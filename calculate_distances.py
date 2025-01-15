def calculate_distances(points):
    from scipy.spatial.distance import pdist, squareform
    import plotly.express as px

    distances_cos = squareform(pdist(points, metric="cosine"))
    print("Минимальное косинусное расстояние:", distances_cos[distances_cos > 0].min())
    print("Среднее косинусное расстояние:", distances_cos.mean())
    print("Дисперсия косинусного расстояния:", distances_cos.var())
    print("Максимальное косинусное расстояние:", distances_cos.max())

    distances_euc = squareform(pdist(points, metric="euclidean"))
    print()
    print("Минимальное евклидово расстояние:", distances_euc[distances_euc > 0].min())
    print("Среднее евклидово расстояние:", distances_euc.mean())
    print("Дисперсия евклидово расстояния:", distances_euc.var())
    print("Максимальное евклидово расстояние:", distances_euc.max())

    px.imshow(distances_cos).show()


def optimize_coulomb_energy(
    N, d, lr=0.05, n_iter=48, verbose=True, log_iter=2, seed=None
):
    """
    Функция для оптимизации N точек на единичной сфере в d-мерном пространстве.
    Стремится к равномерному распределению точек на сфере, имитируя кулоновское
    отталкивание между точками.
    """
    import torch

    # Задаём случайный сид
    if seed is not None:
        torch.manual_seed(seed)

    # Инициализация N случайных векторов из равномерного распределения в [-1, 1]^d
    points = torch.rand((N, d)) * 2 - 1
    points = points / points.norm(dim=1, keepdim=True)
    points.requires_grad_()

    # Функция для вычисления суммарной энергии кулоновского отталкивания
    def coulomb_energy(points):
        N = points.shape[0]
        energy = 0.0

        # Считаем потенциал между всеми парами точек
        for i in range(N):
            for j in range(i + 1, N):
                # Евклидово расстояние между точками i и j
                dist = torch.mean((points[i] - points[j]) ** 2)

                # Избегаем деления на ноль: добавляем очень маленькое значение eps
                eps = 1e-10
                energy += 1.0 / (dist + eps)

        return energy

    # Создаем оптимизатор (SGD)
    optimizer = torch.optim.SGD([points], lr=lr)

    # Основной цикл оптимизации
    for iter in range(n_iter):
        optimizer.zero_grad()  # обнуляем градиенты

        energy = coulomb_energy(points)  # вычисляем энергию системы

        # Для контроля можно выводить энергию каждые, например, 100 итераций
        if verbose and (iter + 1) % log_iter == 0:
            print(f"Iter {(iter + 1):4d} | Energy = {energy.item():.4f}")

        energy.backward()
        optimizer.step()

        # После шага оптимизации проецируем точки обратно на единичную сферу
        with torch.no_grad():
            points.copy_(points / points.norm(dim=1, keepdim=True))

    return points.detach().numpy(), energy
