In this section I show the parameters I set in each section
1. visualisition:
```
min_samples_per_label =20
max_samples_per_label = 500
```
2. linear probing:
```
for epoch in range(40)
```
3. Random forest:
```
rf_classifier = RandomForestClassifier(
    n_estimators=3,
    criterion='gini',
    max_depth=40,
    min_samples_split=20,
    min_samples_leaf=20,
    max_features=0.5,
    max_leaf_nodes=10000,
    bootstrap=False,
    n_jobs=-1
)
```
4. Resnet-18
```
total_samples = 15000
batch_size = 1000
stopvalue = 10
```
