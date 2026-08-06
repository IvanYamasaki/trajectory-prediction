# Mes 10 — Independencia do stream 2019 (validacao da FAR)

n = 48,000 trajetorias (2019, Seq2Seq 30->15), 6 jogos.

## (1) Autocorrelacao e tamanho amostral efetivo

| Sinal | rho(1) | rho(10) | rho(100) | n_eff | n_eff/n | corte |
|-------|-------:|--------:|---------:|------:|--------:|------:|
| ADE bruto | 0.556 | 0.105 | 0.044 | 1,208 | 2.5% | lag 500 |
| ADE robz_w200 | 0.535 | 0.068 | -0.003 | 7,762 | 16.2% | lag 41 |

## (2) IC de Wilson da FAR (k=2 alarmes) com n vs n_eff

| Base | FAR/1k | IC Wilson 95%/1k |
|------|-------:|------------------|
| n = 48,000 | 0.0417 | [0.0114; 0.1519] |
| n_eff = 7,762 (robz) | 0.2577 | [0.0707; 0.9391] |

## (3) Permutacoes do bloco 2019 (B=50, pipeline robz+ADWIN final)

Alarmes observados na ordem real: **2**

| Esquema | k medio | k mediano | [p5; p95] | FAR media/1k | P(k >= obs) |
|---------|--------:|----------:|-----------|-------------:|------------:|
| blocks | 1.84 | 2.0 | [1; 2] | 0.0383 | 0.84 |
| within | 0.00 | 0.0 | [0; 0] | 0.0000 | 0.00 |
| full | 0.00 | 0.0 | [0; 0] | 0.0000 | 0.00 |

Leitura: `blocks` preserva a autocorrelacao intra-jogo (mede o efeito
da ordem dos jogos); `within` destroi a autocorrelacao intra-jogo;
`full` destroi toda a estrutura. Se k(within/full) >> k(obs), a
autocorrelacao intra-jogo NAO inflaciona a FAR observada (ao
contrario: a estrutura local reduz falsos alarmes do robz+ADWIN).