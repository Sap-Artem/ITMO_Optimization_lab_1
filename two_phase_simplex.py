import numpy as np
import argparse

# ===============================
# 🔹 Парсер входного файла
# ===============================
def parse_lp_text(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sense = lines[0].lower()
    c = np.array([float(x) for x in lines[1].split()])
    A, signs, b = [], [], []
    for line in lines[2:]:
        parts = line.split()
        *coeffs, sign, rhs = parts
        A.append([float(x) for x in coeffs])
        signs.append(sign)
        b.append(float(rhs))
    return sense, np.array(c), np.array(A), signs, np.array(b)


# ===============================
# 🔹 Утилиты
# ===============================
def pivot_on(T, row, col):
    """Поворот таблицы по ведущему элементу"""
    pivot = T[row, col]
    T[row, :] /= pivot
    m, n = T.shape
    for r in range(m):
        if r != row:
            T[r, :] -= T[r, col] * T[row, :]


def print_tableau(T):
    np.set_printoptions(precision=4, suppress=True)
    print(np.array_str(T, precision=4, suppress_small=True))


# ===============================
# 🔹 Симплекс-метод
# ===============================
def simplex(T, basis, maximize=True, tol=1e-9, verbose=False, phase_name=""):
    m = T.shape[0] - 1
    n = T.shape[1] - 1
    it = 0
    while True:
        it += 1
        reduced = T[-1, :n]
        if maximize:
            candidates = np.where(reduced > tol)[0]
            if candidates.size == 0:
                if verbose:
                    print(f"\n[{phase_name}] Итерация {it}: оптимум найден")
                return "optimal", T, basis
            e = candidates[np.argmax(reduced[candidates])]
        else:
            candidates = np.where(reduced < -tol)[0]
            if candidates.size == 0:
                if verbose:
                    print(f"\n[{phase_name}] Итерация {it}: оптимум найден")
                return "optimal", T, basis
            e = candidates[np.argmin(reduced[candidates])]

        col = T[:m, e]
        rhs = T[:m, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(col > tol, rhs / col, np.inf)
        if np.all(np.isinf(ratio)):
            print(f"[{phase_name}] Неограниченность по переменной {e}")
            return "unbounded", T, basis
        l = np.argmin(ratio)

        if verbose:
            print(f"\n[{phase_name}] --- Итерация {it} ---")
            print(f"Ведущий столбец (entering): {e}")
            print(f"Ведущая строка (leaving): {l}")
            print("Таблица до пивота:")
            print_tableau(T)
            print("Базис:", basis)

        pivot_on(T, l, e)
        basis[l] = e

        if verbose:
            print("После пивота:")
            print_tableau(T)
            print("Новый базис:", basis)


# ===============================
# 🔹 Двухфазный симплекс-метод
# ===============================
def two_phase(sense, c, A, signs, b, verbose=False):
    m, n = A.shape

    # Преобразуем в равенства (<= → добавляем s, >= → вычитаем s)
    A_eq, b_eq, add_vars = [], [], []
    for i in range(m):
        if signs[i] == "<=":
            row = np.hstack([A[i], np.eye(1, m, i)[0]])
            add_vars.append("slack")
        elif signs[i] == ">=":
            row = np.hstack([A[i], -np.eye(1, m, i)[0]])
            add_vars.append("surplus")
        else:  # "="
            row = np.hstack([A[i], np.zeros(m)])
            add_vars.append("eq")
        A_eq.append(row)
        b_eq.append(b[i])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    n_total = A_eq.shape[1]

    # --- Фаза I ---
    # Проверим, нужны ли искусственные переменные
    need_artificial = any(s in [">=", "="] for s in signs)

    if need_artificial:
        # создаем вспомогательную задачу
        I = np.eye(m)
        A1 = np.hstack([A_eq, I])
        c1 = np.hstack([np.zeros(n_total), np.ones(m)])

        T1 = np.zeros((m + 1, A1.shape[1] + 1))
        T1[:m, :-1] = A1
        T1[:m, -1] = b_eq
        T1[-1, :-1] = c1
        T1[-1, -1] = np.sum(b_eq)
        #    скорректируем последнюю строку, чтобы базис был допустимым
        for i in range(m):
            T1[-1, :] -= T1[i, :]

        basis = list(range(n_total, n_total + m))
        status1, T1_final, basis1 = simplex(
            T1, basis, maximize=True, verbose=verbose, phase_name="Фаза I"
        )

        if verbose:
            print("\n=== Конец Фазы I ===")
            print("Базис:", basis1)
            print("Таблица после Фазы I:")
            print_tableau(T1_final)
            print("====================\n")

        if abs(T1_final[-1, -1]) > 1e-7:
            return {"status": "no feasible solution"}

    else:
        # если только ограничения ≤, базис уже есть
        basis1 = list(range(n, n + m))

    # --- Фаза II ---
    T2 = np.zeros((m + 1, n_total + 1))
    T2[:m, :n_total] = A_eq
    T2[:m, -1] = b_eq
    if sense == "max":
        T2[-1, :n] = c
    else:
        T2[-1, :n] = -c


    status2, T2_final, basis2 = simplex(
        T2, basis1.copy(), maximize=True, verbose=verbose, phase_name="Фаза II"
    )

    if verbose:
        print("\n=== Конец Фазы II ===")
        print("Финальная таблица:")
        print_tableau(T2_final)
        print("Базис:", basis2)
        print("====================\n")

    if status2 == "unbounded":
        return {"status": "unbounded"}

    x = np.zeros(n_total)
    for i, bi in enumerate(basis2):
        if bi < n_total:
            x[bi] = T2_final[i, -1]

    z = -T2_final[-1, -1] if sense == "max" else T2_final[-1, -1]
    return {"status": "optimal", "x": x[:n], "z": z}


# ===============================
# 🔹 Основной запуск
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-phase simplex solver")
    parser.add_argument("file", help="Input LP file")
    parser.add_argument("--verbose", action="store_true", help="Show simplex steps")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        sense, c, A, signs, b = parse_lp_text(f.read())

    res = two_phase(sense, c, A, signs, b, verbose=args.verbose)

    print("\n=== РЕЗУЛЬТАТ ===")
    if res["status"] == "optimal":
        print("Результат: оптимально")
        print("x =", np.round(res["x"], 4))
        print("Значение целевой функции =", np.round(res["z"], 4))
    elif res["status"] == "unbounded":
        print("Функция не ограничена (уходит в +∞)")
    else:
        print("Допустимых решений нет (область X пуста)")
