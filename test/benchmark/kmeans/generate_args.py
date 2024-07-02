observations = [10,100,1000,10000]
features = [10,50,100]
clusters = [2,4,8]
functions=['kmeans_scipy_CPU','kmeans_sklearn_mt_CPU','kmeans_cuml_GPU','kmeans_torch_CPU','kmeans_torch_GPU']

idExec = 0
for func in functions:
    for k in clusters:
        for feature in features:
            for obs in observations:
                    idExec += 1
                    print(f"{idExec} {func} {k} {feature} {obs}")
