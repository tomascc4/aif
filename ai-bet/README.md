# ⚽ OCTO BET

Uma equipa de agentes de AI que analisa os jogos das principais ligas e competições europeias e te entrega um **boletim do dia**: as apostas mais promissoras, cada uma com a odd a partir da qual vale a pena e o porquê, mais uma múltipla sugerida.

Mercados: resultado final (1X2), dupla hipótese, mais/menos 1,5 / 2,5 / 3,5 golos, ambas marcam e handicap asiático.
Competições: Liga Portugal, Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Liga dos Campeões, Liga Europa, Liga Conferência e as taças de Inglaterra, Espanha, Itália, Alemanha e França. Não há Taça de Portugal porque a Odds API não a cobre.

> O mercado de apostas é muito eficiente e as casas têm margem: a maior parte dos sistemas perde. O boletim mostra onde o sistema vê valor, não dá certezas. Aposta só o que estás disposto a perder, e só em casas licenciadas pelo SRIJ.

## Instalação (Mac)

Descomprime para a pasta pessoal (`~/ai-bet`), e não para o Desktop, por causa do iCloud.

```bash
cd ~/ai-bet
./setup.sh
.venv/bin/python -m bet demo            # demo com equipas inventadas → http://localhost:8790
.venv/bin/python -m bet setup-odds      # chave grátis da the-odds-api.com
.venv/bin/python -m bet check           # testa as fontes e o Claude
.venv/bin/python -m bet run             # boletim de hoje
.venv/bin/python -m bet dashboard       # abre o dashboard
./scripts/install_schedule.sh config.toml 10:15     # boletim automático todos os dias às 10:15
./scripts/dashboard_service.sh          # dashboard sempre ligado
```

## O que aparece no dashboard

- **Boletim**: as apostas do dia (até 5), as mais prováveis e uma múltipla.
  - *Valor*: há uma odd real acima da justa. É a única que, em teoria, dá lucro a longo prazo.
  - *Sólida*: provável (55%+) e o sistema (xG, modelos, notícias) concorda com o mercado ou está acima dele.
  - *As mais prováveis*: 72%+ de probabilidade (ex.: Sporting a 1.20 com 80%). Acertam muito, mas só compensam se a odd for pelo menos a indicada.
- **Previsões**: todos os jogos dos próximos 7 dias, com % de vitória, empate e derrota, o resultado mais provável, mais de 2,5 golos e ambas marcam. Cada jogo abre com todos os mercados, a classificação, o xG recente, os confrontos diretos e as notícias.
- **Historial**: o que bateu certo, por tipo de aposta.

## Como se lê uma aposta do boletim

```
★★☆  Liga Portugal · sáb 20:30
Benfica – Porto
Mais de 2,5 golos                       2.10   ← melhor odd encontrada (e em que casa)
58% de ganhar · mercado 54% · justa 1.72  mín. 1.77   ← abaixo disto já não vale a pena
"Os dois marcaram em 5 dos últimos 6..."
```

- **Justa**: a odd com valor esperado zero segundo o sistema.
- **Mínima**: a odd a partir da qual há pelo menos 3% de valor. Se a tua casa pagar menos, salta essa aposta.
- **Estrelas**: tamanho do valor, se os modelos e as notícias apoiam a aposta, e o que disse o advogado do diabo. Apostas pouco prováveis (abaixo de 40%) e odds estimadas ficam no máximo com 2 estrelas.
- **Odd estimada**: quando não há odds reais para aquele mercado, o sistema usa a odd típica de uma casa. Confirma sempre na tua.

## Como funciona

```
Recolha   Arquivista  resultados, remates enquadrados e odds de fecho das 7 ligas (football-data.co.uk, grátis)
          Corretor    odds de várias casas para todas as competições (The Odds API)
          ClubElo     força dos clubes entre ligas diferentes (jogos europeus)
          Understat   xG jogo a jogo das 5 grandes ligas (qualidade das oportunidades)
Modelos   golos esperados = mercado (70%) + xG do Understat (15%) + Dixon-Coles, remates e Elo
          → preço de TODOS os mercados a partir da matriz de resultados exatos
Notícias  Radar       uma pesquisa por competição varre os jogos todos
          Médico      lesões e castigos (pesquisa web, com fontes)
          Repórter    rotação, motivação, treinador, tempo
          Analista    notícias → ajuste limitado aos golos esperados (3 palpites, fica a mediana)
Tribunal  Advogado do Diabo  ataca cada aposta candidata
Boletim   Presidente  escolhe as simples e a múltipla e escreve o porquê
```

Só vão à análise a fundo (onde se gasta quota do Claude) os jogos das próximas 36 horas com valor à partida ou notícia forte, até 8 por dia.

## Quota da Odds API

O plano grátis tem 500 créditos por mês. Cada corrida gasta 2 créditos por competição com jogos nos próximos 3 dias, mais 2 por jogo analisado a fundo (ambas marcam + handicap). Uma corrida por dia ronda os 300-450 créditos por mês. O dashboard mostra os créditos que restam, e o sistema para de gastar quando chegam a 40. Se quiseres mais corridas, o plano de 20k créditos custa cerca de 30 $/mês.

## Historial

Cada boletim fica guardado e fecha-se sozinho quando os jogos acabam:
- acerto real vs o que o sistema esperava;
- lucro "a 1 unidade" (se tivesses apostado 1 em cada);
- **CLV** (a odd apanhada vs o fecho da Pinnacle), que é o sinal de qualidade que aparece mais cedo;
- o mesmo partido por mercado e por estrelas, e se as notícias melhoram as probabilidades.

## Literatura usada

Dixon & Coles (1997) · Constantinou & Fenton (2013, pi-ratings) · Štrumbelj (2014) e Shin (1993) para tirar a margem · Kaunitz et al. (2017) · Hubáček et al. (2019) · Walsh & Joshi (2024, calibração > accuracy) · Halawi et al. (2024, LLMs + pesquisa) · Schoenegger et al. (2024, agregar LLMs) · benchmarks do Mundial 2026 (os LLMs sozinhos não batem o mercado).
