# Artigo SBR 2026 — pacote de regras e template

Tudo que é preciso para escrever e submeter um artigo ao
**SBR 2026 — Brazilian Symposium on Robotics** (João Pessoa/PB, 24–27/11/2026),
em **inglês**, **6 páginas**, formato **IEEE Conference**, revisão **double-blind**.

| Arquivo | Para quê |
|---|---|
| **[GUIA-SBR2026-ARTIGO.md](GUIA-SBR2026-ARTIGO.md)** | Documento principal: todas as regras, datas, formato, double-blind, JEMS, camera-ready. **Comece por aqui.** Inclui na §10 um prompt pronto para outra IA escrever o artigo. |
| **[main.tex](main.tex)** | Esqueleto LaTeX já anonimizado e comentado. Compila sozinho (figuras como placeholder). |
| **[references.bib](references.bib)** | Modelo de bibliografia no padrão IEEE + tabela de abreviações de veículos de robótica. |
| **[CHECKLIST-SUBMISSAO.md](CHECKLIST-SUBMISSAO.md)** | Checklist final antes de clicar em "Submit". |
| **[check_double_blind.ps1](check_double_blind.ps1)** | Varre `.tex`/`.bib`/PDF procurando vazamentos de identidade. |

## Uso rápido

```powershell
# 1. escrever o conteudo em main.tex
# 2. compilar
pdflatex main; bibtex main; pdflatex main; pdflatex main
# 3. checar anonimato
powershell -ExecutionPolicy Bypass -File .\check_double_blind.ps1 -Path . -Pdf .\main.pdf
# 4. renomear para sbr2026_paper.pdf e submeter em https://jems3.sbc.org.br/sbr2026
```

## Prazos

| Marco | Data |
|---|---|
| Submissão | **30/07/2026** |
| Notificação | 15/08/2026 |
| Camera-ready | 15/09/2026 |
| Evento | 24–27/11/2026 |
