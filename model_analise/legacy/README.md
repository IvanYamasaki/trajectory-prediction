# legacy/ — código em quarentena

Scripts que dependem do **contrato antigo de 5 features** por janela
(o pipeline atual usa 6: x, y, vx, vy, sin, cos) e/ou de artefatos que não
existem mais no repositório. Estão aqui apenas como referência histórica —
**não foram consertados** e não devem ser usados.

- **compare_models.py** — quebrado: chama `loader.load_params('dataset/norm_params')`
  (método/arquivo inexistentes no `LoadDataSet` atual), instancia
  `RobotOnlyPredictor` com assinatura posicional errada e faz `build` com
  5 features (`(None, look_back, 5)`).
- **comparison_tests.py** — usado apenas por `compare_models.py`. Além do
  contrato de 5 features no MLP, o `TestLoss` usa `np.sum` para agregar o
  erro, produzindo um "ADE" numa escala incomparável com as métricas
  atuais (mm por passo).
- **batch_logs.py** (ex-`ai_model/batch_logs.py`) — callback `BatchLogs`
  nunca importado por nenhum módulo; `save_vars` grava em
  `'../saved_variables/'`, diretório que não existe.

Única alteração feita ao mover: `parents[1]` → `parents[2]` no preâmbulo de
`compare_models.py`/`comparison_tests.py`, para que o `PROJECT_ROOT` continue
apontando para a raiz do repo a partir deste subdiretório.
