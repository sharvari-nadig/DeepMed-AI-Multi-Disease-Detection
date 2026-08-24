# Place your trained CNN weights here

Copy your trained weights file into this folder so the app can run **real** inference:

```
models/best_model.weights.h5
```

This is the `best_model.weights.h5` produced by your training notebook
(Cell 12 `ModelCheckpoint`, saved to your Google Drive
`multi_diseases_experiments_csv/` folder).

- The file is **git-ignored** on purpose (it is large).
- If this file is missing, the app still starts but runs in **DEMO MODE**
  (simulated predictions, clearly flagged in the UI) so you can test the
  interface and the LLM layer without the weights.
- The path can be changed via `CNN_WEIGHTS_PATH` in your `.env`.

> The app rebuilds the exact architecture from your notebook (Cell 10) and then
> calls `model.load_weights(...)`, so the weights load without needing the full
> `.keras` model file.
