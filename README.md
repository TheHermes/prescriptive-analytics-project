# Preskriptiv Analytik Projekt - Hermes

## Lunar Lander

I detta projekt tänker jag lösa 'LunarLander' miljön med två metoder: DQN (Deep Q Network) och PPO (Proximal Policy Optimization).

Issues installing box2d on windows, maybe colab instead or rahti notebooks

Testing with colab

Vi måste anpassa våra parametrar, episoder, steps mm. för att skapa en bra grund för hitta en bra policy per metoder (DQN, PPO).

Vi använder färdigt implementerade bibliotek med DQN och PPO, via stable_baselines3

### DQN - Deep Q Network

DQN är ett värde baserat reinforcement learning algoritm.
Baserad på Q-learning, men istället för att använda Q-tabellen använder vi ett neuralt nätverk istället. Passar sig bättre för simplare miljöer med diskreta handlingar.

### PPO - Proximal Policy Optimization

PPO är en policybaserad metod inom reinforcement learning. Den lär sig direkt en policy i gemförelse med DQN som beräkndar Q-värden.

PPO förbättrar sig effektivare genom att begränsa hur policyn ändrar sig vid varje uppdatering. Detta skapar en stabil och pålitlig inlärning.

### Resultat

#### LunarLander med DQN

Klarar sig inte så bra, svårt att lösa med DQN: vad är det bästa jag kan göra här?

```Python
dqn_model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=0,
    policy_kwargs=dict(net_arch=[64, 64]),
)

dqn_model.learn(total_timesteps=2_000_000, callback=eval_callback)

# Bättre
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


```Python
Mean reward: 154.8678681 +/- 80.60501831093475

# Bättre
Mean reward: 252.92582385000006 +/- 45.51575735288412
```

#### Visuel Test

![DQN körning](/assets/dqn_lunarlander_run.gif)

Inte så värst bra, når inte 200+ mean reward

#### LunarLander med PPO

Passar sig mycket bättre för lunar lander, eftersom det är en mera komplicerad miljö som PPO är bättre på och pga att den är policy baserad: hur ska agenten agera beroende på sin situation.

800 000 timesteps räcker för träning

#### Parametrar:

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

ppo_model.learn(total_timesteps=1_500_000, callback=eval_callback)
```

Efter inträning:

```Python
Mean reward: 252.22737134999997 +/- 49.11030609688438
```

#### Testande hur modellen klarar sig

![Gif av en körning](/assets/ppo_lunarlander_run.gif)

Inte helt perfekt men den klarar sig nog ofta att landa!

#### Gemförelse
