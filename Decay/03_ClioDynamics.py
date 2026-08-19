# =============================================================================
# БЛОК 1: ИМПОРТ БИБЛИОТЕК
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatLogSlider, FloatSlider, fixed
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# БЛОК 2: УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ ПЛОЩАДИ
# =============================================================================
def compute_area(y, x):
    """
    Вычисляет площадь под кривой методом трапеций.
    Работает во всех версиях NumPy, даже где нет np.trapz.
    """
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)


# =============================================================================
# БЛОК 3: ЯДРО МОДЕЛИ (историческая динамика)
# =============================================================================
def historical_dynamics(lambda_old, lambda_new, N0=100, t_max=None, num_points=1000):
    """
    Модель исторической динамики:
    N1 - старые ресурсы (расходуются)
    N2 - новые ресурсы/инновации (генерируются из старых, но устаревают)
    N3 - устойчивое наследие (накапливается)
    """
    if t_max is None:
        min_lambda = min(lambda_old, lambda_new) if min(lambda_old, lambda_new) > 0 else 0.001
        t_max = 6 * np.log(2) / min_lambda

    t = np.linspace(0, t_max, num_points)

    # Старые ресурсы (убывают)
    N1 = N0 * np.exp(-lambda_old * t)

    # Новые ресурсы/инновации (рождаются из старых, устаревают)
    if abs(lambda_old - lambda_new) < 1e-12:
        N2 = N0 * lambda_old * t * np.exp(-lambda_old * t)
    else:
        N2 = N0 * (lambda_old / (lambda_new - lambda_old)) * (np.exp(-lambda_old * t) - np.exp(-lambda_new * t))

    # Устойчивое наследие
    N3 = N0 - N1 - N2
    N3 = np.maximum(N3, 0)

    return t, N1, N2, N3


# =============================================================================
# БЛОК 4: ИСТОРИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ
# =============================================================================
def plot_history(lambda_old, lambda_new, N0=100, log_scale=False):
    """
    Визуализация исторической динамики с культурно-историческими аннотациями.
    """
    t, N1, N2, N3 = historical_dynamics(lambda_old, lambda_new, N0)

    # Определяем режим и историческую аналогию
    if abs(lambda_old - lambda_new) < 1e-12:
        regime = "РЕЖИМ УСТОЙЧИВОГО РАЗВИТИЯ"
        description = "Общество плавно переходит от старых ресурсов к новым.\n"
        description += "Скорость исчерпания старых = скорости устаревания новых.\n"
        description += "Аналогия: классический Рим или династический Китай —\n"
        description += "медленная эволюция без резких скачков."

        example = "📜 Пример: Древний Египет — веками стабильное развитие\n"
        example += "на основе Нила, без резких технологических прорывов."

    elif lambda_old < lambda_new:
        regime = "РЕЖИМ РАВНОВЕСНОГО НАКОПЛЕНИЯ"
        description = "Старые ресурсы расходуются МЕДЛЕННО,\n"
        description += "а новые генерируются БЫСТРО и быстро устаревают.\n"
        description += "Общество работает как 'конвейер инноваций',\n"
        description += "но наследие (N₃) накапливается постепенно."

        example = "🏛️ Пример: Афины V века до н.э. —\n"
        example += "философия, демократия, театр обновляются каждое поколение,\n"
        example += "но стабильной империи не возникает."

    else:  # lambda_old > lambda_new
        # Вычисляем характеристики для подсказок
        t_peak = np.log(lambda_old / lambda_new) / (lambda_old - lambda_new)
        N2_peak = N0 * (lambda_old / (lambda_new - lambda_old)) * (
                    np.exp(-lambda_old * t_peak) - np.exp(-lambda_new * t_peak))

        if t_peak > 0 and t_peak < t[-1]:
            regime = "РЕЖИМ ИННОВАЦИОННОГО ВЗРЫВА"
            description = f"Пик инноваций (максимальная скорость преобразований) —\n"
            description += f"в момент t = {t_peak:.2f} (в условных единицах времени).\n"
            description += "Старые ресурсы быстро сжигаются для создания нового.\n"
            description += "После пика — инерционное доживание."

            example = f"🔥 Пример: Промышленная революция (Англия, XIX в.) —\n"
            example += f"быстрое сжигание угля и лесов → пар, фабрики, империя.\n"
            example += f"Максимум преобразований пришелся на середину процесса.\n"
            example += f"Сегодня — постиндустриальное 'наследие' (N₃ растёт)."
        else:
            regime = "РЕЖИМ МОНОТОННОГО ИСТОЩЕНИЯ"
            description = "Старые ресурсы сжигаются слишком быстро,\n"
            description += "а новые не успевают создаваться.\n"
            description += "Общество деградирует без перехода к новому укладу."

            example = "💀 Пример: Остров Пасхи —\n"
            example += "леса вырублены для статуй,\n"
            example += "инноваций не возникло, цивилизация рухнула."

    # ====== ПОСТРОЕНИЕ ГРАФИКА ======
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # График 1: Динамика ресурсов
    ax1.plot(t, N1, 'b-', linewidth=2.5, label='N₁: Старые ресурсы (традиционные)')
    ax1.plot(t, N2, 'r-', linewidth=2.5, label='N₂: Новые ресурсы (инновации)')
    ax1.plot(t, N3, 'g-', linewidth=2.5, label='N₃: Устойчивое наследие')
    ax1.fill_between(t, 0, N2, alpha=0.2, color='red', label='Объём инноваций (площадь)')

    # Отмечаем максимум N₂ (если есть)
    if lambda_old != lambda_new:
        t_peak = np.log(lambda_old / lambda_new) / (lambda_old - lambda_new)
        if 0 < t_peak < t[-1]:
            N2_peak = N0 * (lambda_old / (lambda_new - lambda_old)) * (
                        np.exp(-lambda_old * t_peak) - np.exp(-lambda_new * t_peak))
            ax1.scatter(t_peak, N2_peak, color='darkred', s=100, zorder=5,
                        label=f'Пик инноваций: t={t_peak:.2f}')

    ax1.set_xlabel('Время (условные единицы)', fontsize=12)
    ax1.set_ylabel('Объём ресурсов/инноваций', fontsize=12)
    ax1.set_title('Динамика ресурсов в историческом процессе', fontsize=14)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    if log_scale:
        ax1.set_yscale('log')

    # График 2: Режим и интерпретация
    ax2.text(0.05, 0.95,
             f"{regime}\n\n{description}",
             transform=ax2.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.axis('off')

    # График 3: Исторический пример
    ax3.text(0.05, 0.95,
             f"ИСТОРИЧЕСКАЯ АНАЛОГИЯ:\n\n{example}\n\n"
             f"Параметры модели:\n"
             f"λ_старые = {lambda_old:.3f} (скорость исчерпания традиций)\n"
             f"λ_новые = {lambda_new:.3f} (скорость устаревания инноваций)\n"
             f"Отношение λ_старые / λ_новые = {lambda_old / lambda_new:.2f}",
             transform=ax3.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # ====== ДОПОЛНИТЕЛЬНАЯ ИНТЕРПРЕТАЦИЯ ======
    print("\n" + "=" * 70)
    print("ИНТЕРПРЕТАЦИЯ КОЛИЧЕСТВЕННЫХ ПОКАЗАТЕЛЕЙ:")
    print("=" * 70)

    # Площадь под N₂ (суммарный объём инноваций) — ИСПРАВЛЕНО
    area_N2 = compute_area(N2, t)
    area_N3 = compute_area(N3, t)

    print(f"📊 Суммарный объём инноваций (площадь под N₂) = {area_N2:.1f} условных единиц.")
    print(f"   Это мера общей 'работы преобразования' общества.")
    print(f"   Чем выше площадь, тем больше было создано нового (открытий, технологий, реформ).")
    print()
    print(f"📊 Накопленное наследие (площадь под N₃) = {area_N3:.1f} условных единиц.")
    print(f"   Это 'капитал цивилизации' — то, что осталось после пика.")
    print()
    print(f"📈 Соотношение инновации/наследие = {area_N2 / area_N3:.2f}")
    if area_N2 > area_N3:
        print("   Общество было 'инновационно-активным': большая часть работы пошла на создание нового.")
    else:
        print("   Общество было 'консервативно-наследственным': большая часть — это накопление и сохранение.")

    # Если есть пик — интерпретируем его значение
    if lambda_old != lambda_new:
        t_peak = np.log(lambda_old / lambda_new) / (lambda_old - lambda_new)
        if 0 < t_peak < t[-1]:
            N2_peak = N0 * (lambda_old / (lambda_new - lambda_old)) * (
                        np.exp(-lambda_old * t_peak) - np.exp(-lambda_new * t_peak))
            print()
            print(f"⏳ Пик инновационной активности: t = {t_peak:.2f}")
            print(f"   В этот момент скорость преобразований была максимальной.")
            print(f"   Максимальная 'мгновенная работа' общества = {N2_peak:.1f} единиц.")

    print("=" * 70)


# =============================================================================
# БЛОК 5: ИНТЕРАКТИВНЫЙ РЕЖИМ (для Jupyter)
# =============================================================================
def interactive_history():
    """Интерактивное исследование исторических сценариев"""

    @interact(
        lambda_old=FloatLogSlider(
            value=0.1, base=10, min=-3, max=1, step=0.1,
            description='λ₁: скорость исчерпания старых ресурсов',
            style={'description_width': 'initial'}
        ),
        lambda_new=FloatLogSlider(
            value=0.05, base=10, min=-3, max=1, step=0.1,
            description='λ₂: скорость устаревания инноваций',
            style={'description_width': 'initial'}
        ),
        log_scale=FloatSlider(
            value=0, min=0, max=1, step=1,
            description='Логарифмическая шкала (0=Нет, 1=Да)',
            style={'description_width': 'initial'}
        )
    )
    def update(lambda_old, lambda_new, log_scale):
        plot_history(lambda_old, lambda_new, N0=100, log_scale=bool(log_scale))

    return update


# =============================================================================
# БЛОК 6: ИСТОРИЧЕСКИЕ СЦЕНАРИИ
# =============================================================================
def historical_scenarios():
    """Пять типовых исторических сценариев"""

    scenarios = {
        'Древний Египет (стабильность)': (0.02, 0.02),
        'Афинская демократия (равновесие)': (0.02, 0.2),
        'Промышленная революция (взрыв)': (0.2, 0.03),
        'Остров Пасхи (коллапс)': (0.5, 0.01),
        'Современный Китай (быстрый рост)': (0.1, 0.001)
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (title, (l1, l2)) in enumerate(scenarios.items()):
        if idx >= len(axes):
            break

        t, N1, N2, N3 = historical_dynamics(l1, l2, N0=100)

        ax = axes[idx]
        ax.plot(t, N1, 'b-', linewidth=2, label='Старые ресурсы')
        ax.plot(t, N2, 'r-', linewidth=2, label='Инновации')
        ax.plot(t, N3, 'g-', linewidth=2, label='Наследие')
        ax.fill_between(t, 0, N2, alpha=0.15, color='red')

        # Отмечаем пик инноваций
        if l1 != l2:
            t_peak = np.log(l1 / l2) / (l1 - l2)
            if 0 < t_peak < t[-1]:
                N2_peak = 100 * (l1 / (l2 - l1)) * (np.exp(-l1 * t_peak) - np.exp(-l2 * t_peak))
                ax.scatter(t_peak, N2_peak, color='darkred', s=80, zorder=5)

        ax.set_title(f'{title}\nλ₁={l1:.2f}, λ₂={l2:.2f}', fontsize=10)
        ax.set_xlabel('Время')
        ax.set_ylabel('Объём')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    for i in range(len(scenarios), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# =============================================================================
# БЛОК 7: ТОЧЕЧНЫЙ АНАЛИЗ
# =============================================================================
def analyze_history(lambda_old, lambda_new, N0=100):
    """Детальный анализ конкретного исторического сценария"""

    print("\n" + "=" * 70)
    print("ИСТОРИЧЕСКИЙ АНАЛИЗ ПО МОДЕЛИ РЕСУРСОВ")
    print("=" * 70)

    t, N1, N2, N3 = historical_dynamics(lambda_old, lambda_new, N0)

    # Определяем ключевые моменты
    if lambda_old != lambda_new:
        t_peak = np.log(lambda_old / lambda_new) / (lambda_old - lambda_new)
        if 0 < t_peak < t[-1]:
            N2_peak = N0 * (lambda_old / (lambda_new - lambda_old)) * (
                        np.exp(-lambda_old * t_peak) - np.exp(-lambda_new * t_peak))
            print(f"📌 ПИК ИННОВАЦИЙ:")
            print(f"   Время: t = {t_peak:.3f} (условных единиц)")
            print(f"   Объём инноваций (максимальная работа): {N2_peak:.1f}% от начальных ресурсов")
            print()

    # Время, когда N₂ = 0.5 * N₂_max (полуширина)
    N2_max = np.max(N2)
    if N2_max > 0:
        idx_half = np.where(N2 >= 0.5 * N2_max)[0]
        if len(idx_half) > 1:
            t_start = t[idx_half[0]]
            t_end = t[idx_half[-1]]
            duration = t_end - t_start
            print(f"📌 ДЛИТЕЛЬНОСТЬ АКТИВНОЙ ФАЗЫ (инновации > 50% от пика):")
            print(f"   От {t_start:.3f} до {t_end:.3f} — длительность {duration:.3f}")
            print(f"   Это 'золотой век' общества, период наиболее интенсивных преобразований.")
            print()

    # Итоговое наследие
    N3_end = N3[-1]
    print(f"📌 ИТОГОВОЕ НАСЛЕДИЕ:")
    print(f"   К концу процесса сохраняется {N3_end / N0:.1%} от начальных ресурсов")
    if N3_end / N0 > 0.5:
        print("   ⭐ Общество оставило богатое наследие, оказавшее влияние на будущие циклы.")
    else:
        print("   ⚠️ Общество мало что оставило после себя — влияние на историю невелико.")

    # Рекомендация по типу общества
    print()
    print(f"📌 ТИП ОБЩЕСТВА:")
    if lambda_old < 0.05 and lambda_new < 0.05:
        print("   🏛️  Традиционное аграрное общество — медленное развитие, устойчивость.")
    elif lambda_old > 0.1 and lambda_new > 0.1:
        print("   🏭  Индустриальное общество — быстрые изменения, короткие циклы.")
    elif lambda_old > 0.1 and lambda_new < 0.05:
        print("   🚀  Инновационное общество с долгим наследием — прорыв, затем стагнация.")
    elif lambda_old < 0.05 and lambda_new > 0.1:
        print("   🔄  Общество быстрых реформ — постоянное обновление без накопления.")
    else:
        print("   🤔  Смешанный тип.")

    print("=" * 70)


# =============================================================================
# БЛОК 8: ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("МОДЕЛЬ ИСТОРИЧЕСКОЙ ДИНАМИКИ: РЕСУРСЫ → ИННОВАЦИИ → НАСЛЕДИЕ")
    print("=" * 70)
    print("\nДоступные режимы:")
    print("  1. Демонстрация 5 исторических сценариев")
    print("  2. Интерактивное исследование (Jupyter)")
    print("  3. Анализ конкретного сценария")
    print("  4. Построение графика для произвольных параметров")
    print("-" * 70)

    # --- 1. Исторические сценарии ---
    print("\n▶ 5 ИСТОРИЧЕСКИХ СЦЕНАРИЕВ:")
    historical_scenarios()

    # --- 2. Пример: Промышленная революция ---
    print("\n▶ ПРИМЕР: ПРОМЫШЛЕННАЯ РЕВОЛЮЦИЯ (Англия, XIX век)")
    plot_history(lambda_old=0.2, lambda_new=0.03, N0=100)
    analyze_history(lambda_old=0.2, lambda_new=0.03)

    # --- 3. Пример: Остров Пасхи (коллапс) ---
    print("\n▶ ПРИМЕР: ОСТРОВ ПАСХИ (ресурсный коллапс)")
    plot_history(lambda_old=0.5, lambda_new=0.01, N0=100)

    # --- 4. Пример: Древний Египет (стабильность) ---
    print("\n▶ ПРИМЕР: ДРЕВНИЙ ЕГИПЕТ (тысячелетняя стабильность)")
    plot_history(lambda_old=0.02, lambda_new=0.02, N0=100)

    # --- Интерактивный режим (для Jupyter) ---
    print("\n" + "=" * 70)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ (для Jupyter):")
    print("Раскомментируйте строку ниже для запуска с ползунками.")
    print("=" * 70)
    # interactive_history()  # <-- Раскомментировать для Jupyter