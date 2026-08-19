"""
переходим к классической задаче ядерной физики — последовательному радиоактивному распаду.
Это уравнение Бейтмана для двух дочерних продуктов, и оно идеально иллюстрирует
ту же методологию "параметризации незнания", которую мы обсуждали:
вы вводите скорости распада λ₁ и λ₂ как свободные параметры и смотрите на эволюцию системы.

Математическая модель
Пусть:
N₁(t) — масса первого вещества (родительского)
N₂(t) — масса второго вещества (дочернего)
N₃(t) — масса стабильного конечного продукта
Скорости распада характеризуются постоянными распада λ₁ и λ₂ (или периодами полураспада T₁ = ln2/λ₁, T₂ = ln2/λ₂).

Система дифференциальных уравнений
dN₁/dt = -λ₁ · N₁
dN₂/dt =  λ₁ · N₁ - λ₂ · N₂
dN₃/dt =  λ₂ · N₂

Аналитическое решение (закон Бейтмана)
При начальных условиях N₁(0) = N₀, N₂(0) = 0, N₃(0) = 0:
N₁(t) = N₀ · e^(-λ₁·t)
N₂(t) = N₀ · (λ₁ / (λ₂ - λ₁)) · (e^(-λ₁·t) - e^(-λ₂·t)), при λ₁ ≠ λ₂
N₃(t) = N₀ · (1 - (λ₂·e^(-λ₁·t) - λ₁·e^(-λ₂·t)) / (λ₂ - λ₁))
Особый случай при λ₁ = λ₂:
N₂(t) = N₀ · λ₁ · t · e^(-λ₁·t)
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatLogSlider, FloatSlider, fixed
from IPython.display import display


def radioactive_decay_chain(lambda1, lambda2, N0=100, t_max=None, num_points=1000):
    """
    Модель последовательного радиоактивного распада:
    N1 -> N2 -> N3 (стабильный)

    Параметры:
    - lambda1: постоянная распада первого вещества (1/время)
    - lambda2: постоянная распада второго вещества (1/время)
    - N0: начальная масса первого вещества
    - t_max: максимальное время моделирования (если None, вычисляется автоматически)
    - num_points: количество точек по времени
    """

    # Автоматический выбор t_max
    if t_max is None:
        # Время, когда оба вещества почти полностью распались (5 периодов самого медленного)
        t_max = 5 * np.log(2) / min(lambda1, lambda2) if min(lambda1, lambda2) > 0 else 10

    t = np.linspace(0, t_max, num_points)

    # Аналитические решения
    N1 = N0 * np.exp(-lambda1 * t)

    # Решение для N2 с учетом особого случая lambda1 == lambda2
    if abs(lambda1 - lambda2) < 1e-12:
        N2 = N0 * lambda1 * t * np.exp(-lambda1 * t)
    else:
        N2 = N0 * (lambda1 / (lambda2 - lambda1)) * (np.exp(-lambda1 * t) - np.exp(-lambda2 * t))

    # N3 из закона сохранения массы
    N3 = N0 - N1 - N2
    # Защита от отрицательных значений (из-за ошибок округления)
    N3 = np.maximum(N3, 0)

    return t, N1, N2, N3


def plot_decay_chain(lambda1, lambda2, N0=100, t_max=None, log_scale=False):
    """Визуализация цепочки распада"""

    t, N1, N2, N3 = radioactive_decay_chain(lambda1, lambda2, N0, t_max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: Массы веществ
    ax1.plot(t, N1, 'b-', linewidth=2.5, label=f'Вещество 1 (λ₁={lambda1:.2e})')
    ax1.plot(t, N2, 'r-', linewidth=2.5, label=f'Вещество 2 (λ₂={lambda2:.2e})')
    ax1.plot(t, N3, 'g-', linewidth=2.5, label='Стабильный продукт (N₃)')
    ax1.fill_between(t, 0, N2, alpha=0.2, color='red')
    ax1.set_xlabel('Время', fontsize=12)
    ax1.set_ylabel('Масса (в долях от N₀)', fontsize=12)
    ax1.set_title('Динамика масс в цепочке распада', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylim([1e-3, N0 * 1.1])

    # График 2: Периоды полураспада и отношения
    T1 = np.log(2) / lambda1 if lambda1 > 0 else np.inf
    T2 = np.log(2) / lambda2 if lambda2 > 0 else np.inf

    # Время максимума N2
    if lambda1 != lambda2:
        t_max_N2 = np.log(lambda1 / lambda2) / (lambda1 - lambda2) if lambda1 != lambda2 else 1 / lambda1
        N2_max = (N0 * lambda1 / (lambda2 - lambda1)) * (np.exp(-lambda1 * t_max_N2) - np.exp(-lambda2 * t_max_N2))
        t_max_N2 = t_max_N2 if t_max_N2 > 0 and t_max_N2 < t[-1] else None
    else:
        t_max_N2 = 1 / lambda1
        N2_max = N0 * lambda1 * t_max_N2 * np.exp(-lambda1 * t_max_N2)

    # Отображение ключевых характеристик
    info_text = f"""
    Ключевые параметры:
    ─────────────────────
    T₁ (период полураспада N₁) = {T1:.3f}
    T₂ (период полураспада N₂) = {T2:.3f}
    Отношение λ₁/λ₂ = {lambda1 / lambda2:.3f}
    Отношение T₁/T₂ = {T1 / T2:.3f}

    Состояние равновесия:
    ─────────────────────
    """

    if t_max_N2 is not None:
        info_text += f"Максимум N₂: t = {t_max_N2:.3f}, масса = {N2_max / N0:.2%} от N₀\n"

    # Вековое равновесие (λ₁ << λ₂)
    if lambda1 < lambda2:
        N2_equilibrium = N0 * lambda1 / lambda2
        info_text += f"Вековое равновесие: N₂ ≈ {N2_equilibrium / N0:.2%} от N₀ (при t→∞)"
    elif lambda1 > lambda2:
        info_text += "Режим: N₂ распадается быстрее, чем образуется (переходный максимум)"
    else:
        info_text += "Режим: λ₁ = λ₂ — вырожденный случай"

    ax2.text(0.1, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    return fig, (t, N1, N2, N3)


# ============================================================================
# ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС (для Jupyter Notebook)
# ============================================================================

def interactive_decay():
    """
    Запускает интерактивный виджет для исследования модели.
    Использует логарифмические ползунки для широкого диапазона параметров.
    """

    @interact(
        lambda1=FloatLogSlider(
            value=0.1, base=10, min=-3, max=2, step=0.1,
            description='λ₁ (1/время)',
            style={'description_width': 'initial'}
        ),
        lambda2=FloatLogSlider(
            value=0.05, base=10, min=-3, max=2, step=0.1,
            description='λ₂ (1/время)',
            style={'description_width': 'initial'}
        ),
        N0=fixed(100),
        log_scale=FloatSlider(
            value=0, min=0, max=1, step=1,
            description='Логарифмическая шкала',
            style={'description_width': 'initial'}
        )
    )
    def update(lambda1, lambda2, N0, log_scale):
        # Автоматический подбор t_max на основе самых медленных процессов
        t_max = 8 * np.log(2) / min(lambda1, lambda2) if min(lambda1, lambda2) > 0 else 20
        plot_decay_chain(lambda1, lambda2, N0, t_max, bool(log_scale))

    return update


# ============================================================================
# РЕЖИМЫ РАБОТЫ (пресеты для быстрого анализа)
# ============================================================================

def demonstrate_scenarios():
    """Демонстрация различных сценариев распада"""

    scenarios = {
        'Вековое равновесие (λ₁ << λ₂)': (0.01, 0.2),
        'Переходный режим (λ₁ > λ₂)': (0.2, 0.05),
        'Вырожденный случай (λ₁ = λ₂)': (0.1, 0.1),
        'Очень медленный N₁, быстрый N₂': (0.001, 0.5),
        'Быстрый N₁, очень медленный N₂': (0.5, 0.001)
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (title, (l1, l2)) in enumerate(scenarios.items()):
        if idx >= len(axes):
            break

        t, N1, N2, N3 = radioactive_decay_chain(l1, l2, N0=100)

        ax = axes[idx]
        ax.plot(t, N1, 'b-', linewidth=2, label='N₁')
        ax.plot(t, N2, 'r-', linewidth=2, label='N₂')
        ax.plot(t, N3, 'g-', linewidth=2, label='N₃')
        ax.fill_between(t, 0, N2, alpha=0.15, color='red')

        ax.set_title(f'{title}\nλ₁={l1:.3f}, λ₂={l2:.3f}', fontsize=10)
        ax.set_xlabel('Время')
        ax.set_ylabel('Масса')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    # Убираем пустые графики
    for i in range(len(scenarios), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# ============================================================================
# АНАЛИЗ ВРЕМЕННЫХ ХАРАКТЕРИСТИК
# ============================================================================

def analyze_equilibrium(lambda1, lambda2):
    """Анализ времени установления равновесия и массы N₂ в равновесии"""

    T1 = np.log(2) / lambda1
    T2 = np.log(2) / lambda2

    print("=" * 60)
    print("АНАЛИЗ ЦЕПОЧКИ РАДИОАКТИВНОГО РАСПАДА")
    print("=" * 60)
    print(f"λ₁ = {lambda1:.4e}, T₁ = {T1:.4f}")
    print(f"λ₂ = {lambda2:.4e}, T₂ = {T2:.4f}")
    print(f"Отношение λ₁/λ₂ = {lambda1 / lambda2:.4f}")
    print("-" * 60)

    if lambda1 < lambda2:
        # Вековое равновесие
        N2_eq = 100 * lambda1 / lambda2
        t_eq = 5 * T2  # Время установления равновесия (~5 периодов дочернего)
        print("РЕЖИМ: Вековое равновесие")
        print(f"Масса N₂ в равновесии: {N2_eq:.2f}% от N₀")
        print(f"Время установления равновесия: t ≈ {t_eq:.2f} (≈ 5·T₂)")
        print(f"Активность N₂ ≈ активности N₁: λ₂·N₂ ≈ λ₁·N₁")

    elif lambda1 > lambda2:
        # Переходный режим
        t_max_N2 = np.log(lambda1 / lambda2) / (lambda1 - lambda2)
        N2_max = 100 * (lambda1 / (lambda2 - lambda1)) * (np.exp(-lambda1 * t_max_N2) - np.exp(-lambda2 * t_max_N2))
        print("РЕЖИМ: Переходный (N₂ накапливается и медленно распадается)")
        print(f"Максимум N₂: t_max = {t_max_N2:.4f}, масса = {N2_max:.2f}% от N₀")
        print("N₂ накапливается быстрее, чем распадается до определенного момента")

    else:
        # Вырожденный случай
        print("РЕЖИМ: Вырожденный (λ₁ = λ₂)")
        t_max_N2 = 1 / lambda1
        N2_max = 100 * lambda1 * t_max_N2 * np.exp(-1)
        print(f"Максимум N₂: t_max = {t_max_N2:.4f}, масса = {N2_max:.2f}% от N₀")
        print(f"Выражение N₂(t) = N₀·λ·t·e^(-λt)")

    print("=" * 60)

    # Дополнительно: время достижения максимума N₂
    if lambda1 != lambda2:
        t_max = np.log(lambda1 / lambda2) / (lambda1 - lambda2)
        if t_max > 0:
            print(f"Время максимума N₂: t = {t_max:.4f}")


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    # 1. Интерактивный режим (для Jupyter)
    print("Запуск интерактивного режима...")
    print("Используйте ползунки для изменения λ₁ и λ₂")
    # interactive_decay()  # Раскомментировать для Jupyter

    # 2. Демонстрация сценариев
    demonstrate_scenarios()

    # 3. Анализ конкретного примера
    print("\n" + "=" * 60)
    print("ПРИМЕР: Анализ цепочки распада")
    print("=" * 60)
    analyze_equilibrium(lambda1=0.1, lambda2=0.02)

    # 4. Построение графика для конкретных параметров
    plot_decay_chain(lambda1=0.1, lambda2=0.02, N0=100)

