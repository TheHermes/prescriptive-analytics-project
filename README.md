# Preskriptiv Analytik Projekt - Hermes

## Lunar Lander

Syftet med projektet är med hjälp av reinforcement learning träna agentbaserade modeller att landa ett rymdfarkost i OpenAI gym miljön Lunar Lander. Fokuset ligger i att lösa miljön med två RL-metoder DQN (Deep Q Network) och PPO (Proximal Policy Optimization), träna dom, analysera och jämföra resultaten. Med lösa menar jag egentligen skapa den bästa modellen jag kan.

Lunar Lander är en miljö var en agent ska landa ett rymdfarkost ner i ett markerat område utan att krascha och innan episoden är slut. [Länk till sida om LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

Varje modell tränas i 2 miljoner steg och under träningen använder vi oss av en eval_callback funktion för att verifiera hur bra inträningen går samt spara den bästa modellen.

Vi måste anpassa våra parametrar, episoder, steps mm. för att skapa en bra grund för hitta en bra policy per metoder (DQN, PPO).

Vi använder färdigt implementerade bibliotek för DQN och PPO, via stable_baselines3 för att träna och utvärdera modellerna.

Vi jobbar i google colab för att träna modellerna samt visualisera modellernas inträning. Lokalt testar, filmar och evaluerar vi modellerna. Jag inkluderar också ipynb filen från colab om inte colab fungerar. Lokalt körde jag med anaconda, eftersom det var ända sättet att köra box2d miljön på windows problemfritt. [Gymnasium paketet för anaconda jag använde.](https://anaconda.org/conda-forge/gymnasium-box2d)

### Tränings kod

Jag tränade modellerna i colab.

[Google Colab Länk](https://colab.research.google.com/drive/1brESnGEeAx9zn20RFZzVxKWi-MrcQixH?usp=sharing)

## Resultat

Båda modellerna finns i varsin training katalog i logs under namnet best_model.zip
I training katalogen finns också en modell av den sista tränade modellen. Vi använde best_model för evaluering, testning visuellt och inspelning.

### DQN - Deep Q Network

DQN är ett värde baserat reinforcement learning algoritm.
Baserad på Q-learning, men istället för att använda Q-tabellen använder vi ett neuralt nätverk istället. Passar sig bättre för simplare miljöer med diskreta handlingar. Kanske inte den bästa för Lunar Lander men vi testar hur det går.

### LunarLander med DQN

#### Tränings visualisering

![DQN Träning](/assets/dqn_training.png)

Träningen är lite hackig och hittar aldrig en smidig träning. Vi når ändå en genomsnittlig belöning på över 200 mot slutet.

Parametrarna hittades genom testande med optuna, informations sökning och AI stöd.

```Python
dqn_model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-5,
    buffer_size=200_000,
    learning_starts=20_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0,
)

dqn_model.learn(total_timesteps=2_000_000, callback=eval_callback)
```

### DQN Modell Evaluering från colab

```Python
Mean reward: 263.33771715 +/- 38.206357018073966
```

### Visuel Test

![DQN körning](/assets/dqn_lunarlander_run_2.gif)

Klarar sig bra inte helt perfekt men löser det ofta!

### PPO - Proximal Policy Optimization

PPO är en policybaserad metod inom reinforcement learning. Den lär sig direkt en policy i jämförelse med DQN som beräknar Q-värden.

PPO förbättrar sig effektivare genom att begränsa hur policyn ändrar sig vid varje uppdatering. Detta skapar en stabil och pålitlig inlärning, vilket hindrar stora och drastiska ändringar till policyn.

### LunarLander med PPO

#### Tränings visualisering

![PPO Träning](/assets/ppo_training.png)

Träningen med PPO är mycket smidigare och snabbare. Dess belöningar stiger snabbt till höga nivåer men sedan når den inte längre efter en stund, 2 miljoner steg är helt för mycket träning.

Passar sig mycket bättre för lunar lander, eftersom det är en mera komplicerad miljö som PPO är bättre på och pga att den är policy baserad: hur ska agenten agera beroende på sin situation.

800 000 timesteps räcker för träning.

### Parametrar:

```Python
ppo_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=0,
)

ppo_model.learn(total_timesteps=2_000_000, callback=eval_callback)
```

### PPO Modell Evaluering

```Python
Mean reward: 281.89231415000006 +/- 32.58030784277455
```

### Testande hur modellen klarar sig

![Gif av en körning](/assets/ppo_lunarlander_run_2.gif)

Inte helt perfekt men den klarar sig nog att landa!

### Jämförelse och Slutsats

|Modell|Mean Reward|Std|Timesteps|Kommentar|
|------|-----------|---|---------|---------|
|DQN|263.3|±38.2|2M|Ostadig inlärning, lär sig långsamt, borde tränas även längre, kanske även inte använda för denna miljö|
|PPO|281.9|±32.6|2M (800k räcker)|Snabb och stadig inlärning, når sitt bästa snabbt, passar välidgt bra för denna miljö|

PPO presterar betydligt mycket bättre än DQN i denna miljö. Den når snabbare högre belöningar och når bättre resultat. DQN lär sig, men långsammare och ostadigt.

PPO fungerar bättre, eftersom den uppdaterar policyn genom att klippa gradienterna (clip range) och skapa en stabilare ändring i policyn. Detta förhindrar stora ändringar i policy och gör den mera effektiv. DQN däremot är känslig för hyperparametrar och fungerar sämre i kontinuerliga tillståndsrum. DQN kämpar med stabilitet och borde antingen ändra på parametrarna eller träna ännu mera.

Både DQN och PPO metoderna lyckades med att landa farkosten, men PPO visade sig vara överlägset bättre i sin prestanda. PPO lär sig stabilare och snabbare medan DQN inte lär sig stabilt och det tar längre tid att blir bättre. Valet av metoden oftast är det viktigaste när man väljer en metod för att träna för att få bra resultat.

Länkar som använts:

[DQN Stable baselines](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)

[PPO Stable baselines](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
