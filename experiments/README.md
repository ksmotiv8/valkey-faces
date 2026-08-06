# Experiments

Reproduce the HNSW vs FLAT study in [../EXPERIMENTS.md](../EXPERIMENTS.md).

```bash
pip install numpy redis
docker run -d --name vbench --rm -p 16401:6379 valkey/valkey-bundle
python3 bench.py     # latency crossover + EF_RUNTIME sweep
python3 recall.py    # recall@1 vs right-person, well-populated clusters
docker rm -f vbench
```

Both scripts talk to port 16401 so they never touch a demo instance on
6379. Numbers are machine-dependent; the shapes are the point.
