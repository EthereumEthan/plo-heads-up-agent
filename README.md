# plo-heads-up-agent

Heads-up short-deck PLO agent for the TAMU Datathon poker challenge. `agent.py` is the organizers'
starter server; the bot itself is the `pokerbot/` package (Monte Carlo equity, tunable thresholds,
opponent model) wired in through `submission/`.

Run locally: `pip install -r requirements.txt && python agent.py`.
