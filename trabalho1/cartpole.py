# UNIVERSIDADE FEDERAL DA FRONTEIRA SUL - UFFS, Campus Chapecó.
# Ciência da Computação, 7a Fase, Matutino.
# Trabalho 01 - CartPole usando Q-Learning.
# Estudantes: João Marcos Campagnolo e Renan Casaca.

# Imports:
import gym # Biblioteca contendo o prblema do Cart-Pole.
import numpy as np
import matplotlib.pyplot as plt
import random
import os.path

# Inicializando o Ambiente do Cart-Pole:
env = gym.make('CartPole-v0')

# Opções do Programa:
LOAD_DICT = True # Habilite para carregar os estados previamente salvos.
PLOT_ALL_EPS = True # Habilite para exibir graficamente todos os Episódios.
PRINT_DICT = False # Habilite para printar a configuração final do Dicionário de Estados.
MAX_STATES = 10**4 # Número de estados que serão gerados. (Multiplicação de todas as possibilidades. <intervalDist>).
MAX_ACTIONS = 200 # Máximo de ações que o carrinho pode fazer dentro de um Episódio.
DESC = 0.9 # Desconto.
ALPHA = 0.01 # Learning Rate
EXPLORE_START = 1.0 # O valor inicial de Exploração. (1.0 significa 100%).
EXPLORE_RATE = 0.9995 # O valor que multiplica a Taxa de Exploração a cada novo Episódio executado.

# Função para percorrer o dicionário e retornar a chave do maior valor e o maior valor.
def maxDict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v # Retorna o indice (key) e o valor.

# Função de criação dos intervalDist <buckets>:
def createIntervalDists():
    intervalDist = np.zeros((4,10)) # Divide o espaço linear de cada dôminio em 10 intervalos.
    intervalDist[0] = np.linspace(-4.8, 4.8, 10) # Posição do Carrinho: [-4.8 a 4.8].
    intervalDist[1] = np.linspace(-5, 5, 10) # Velocidade do Carrinho: [-inf a inf]. Fixado como [-5 a 5].
    intervalDist[2] = np.linspace(-.418, .418, 10) # Angulo do Pêndulo: [-41.8 a 41.8].
    intervalDist[3] = np.linspace(-5, 5, 10) # Velocidade do Pêndulo: [-inf a inf]. Fixado como [-5 a 5].
    return intervalDist

# Função que retorna o estado associado à variável de observação e o bin passados por parâmetros.
def assignBins(observation, intervalDist):
    state = np.zeros(4) # Cria um array de 4 posições para o estado.
    for i in range(4):
        state[i] = np.digitize(observation[i], intervalDist[i]) # Retorna em qual intervalo do intervalDist está o valor da variável passada por parâmetro.
        # Exemplo: para observation[i] = -3.2 e intervalDist[i] = [-5, -3.9, -2.8, -1.7, -0.6, 0.6, 1.7, 2.8, 3.9, 5], o retorno será: 2.
    return state # Exemplo de Estado: state = [5, 6, 6, 5].

# Função para transformar um array de Estado em String:
def getStateAsString(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state # Exemplo: Para state = [5, 6, 6, 5], o retorno será: "5665".

# Função que retorna um array com todos os Estados em forma de String:
def getAllStatesAsString():
    states = []
    for i in range(MAX_STATES):
        states.append(str(i).zfill(4)) # Cada estado será formato para 4 algarismos. Exemplo: Estado 15 -> 0015, Estado 123 -> 0123.
    return states

# Função que instância o Dicionário de Estados:
def initializeQ():
    Q = {} # Inicializa como vazio.
    all_states = getAllStatesAsString() # Adiciona todos os estados ao Dicionário.
    for state in all_states:
        Q[state] = {} # Cada estado do Dicionário recebe uma lista vazia.
        for action in range(env.action_space.n): # Para cada ação possível (Neste problema são duas: Mover para Esquerda ou Direita):
            Q[state][action] = 0 # Inicializa ambas as ações com o valor 0.
    return Q

# Função que conta quantos Estados foram visitados:
def countStates(q):
    visited = 0 # Contador de Estados já visitados.
    all_states = getAllStatesAsString()
    for state in all_states:
        if q[state][0] != 0 or q[state][1] != 0: # Se uma das ações possuir um valor:
            visited += 1 # Incrementa o número de Estados já visitados.
    return visited

# Função para executar um Episódio:
def runEpisode(intervalDist, Q, explore):
    observation = env.reset() # Reinicia as variáveis do Ambiente Observado.
    done = False # Variável que controla o fim do Episódio.
    cnt = 0 # Variável que conta o número de ações realizadas no Episódio.
    state = getStateAsString(assignBins(observation, intervalDist)) # O Estado atual em que o Ambiente se encontra.
    total_reward = 0
    while not done: # Enquanto o número de ações realizadas for menor que o máximo definido e o Pêndulo não cair a baixo do limite:
        cnt += 1 # Incrementa uma Ação.
        # Realização das Ações:
        if np.random.uniform() < explore: # Se for um Episódio Exploratório:
            act = env.action_space.sample() # Toma uma Ação aleatória.
        else: # Se não for um Episódio Exploratório:
            if Q[state][0] == 0 and Q[state][1] == 0: # Se a Recompensa de ambas as Ações possíveis estiverem zeradas:
                act = env.action_space.sample() # Toma uma Ação aleatória.
            else: # Senão estiverem zeradas e não for Exploração:
                act = maxDict(Q[state])[0] # Toma a ação com melhor Recompensa.
        # Opção de Renderização do Ambiente:
        if PLOT_ALL_EPS and not explore: # Pode ser desabilitada para fins de treinamento utilizando menos processamento.
            env.render()
        # Tomando uma Ação:
        # Para cada vez que uma Ação é tomada, as Variáveis de Ambiente são atualizadas, assim como a Recompensa e a flag de Falha.
        observation, reward, done, _ = env.step(act) # Sempre que uma Ação é executada com sucesso, a Recompensa é +1.
        total_reward += reward # A Recompensa total é atualizada.
        # Definindo uma Recompensa de Penalidade para quando o Episódio terminar com o Pêndulo caindo:
        if done and cnt < MAX_ACTIONS: # Se o número de Ações realizadas for menor que o máximo permitido e o Pêndulo caiu:
            reward = - (MAX_ACTIONS*2) # Definindo uma Recompensa negativa.
        if cnt == MAX_ACTIONS:
            done = True
        # Criando um novo Estado a partir das novas Variáveis de Ambiente:
        state_new = getStateAsString(assignBins(observation, intervalDist))
        # Atualizando os valores do Estado:
        a1, max_q_s1a1 = maxDict(Q[state_new]) # Retorna o índice e valor da melhor ação, respectivamente.
        Q[state][act] += ALPHA*(reward + DESC*max_q_s1a1 - Q[state][act]) # Fórmula do Q-Learning.
        state, act = state_new, a1 # Atualizando o Estado Atual e a Ação a ser tomada.
    return total_reward, cnt # Retorna a Recompensa Final do Episódio e o número de Ações Executadas.

# Função para executar vários Episódios:
def runNEpisodes(intervalDist, explore, N=10000):
    # Criação do Dicionário de Estados:
    if LOAD_DICT: # Se quisermos carregar os estados previamente salvos:
        if os.path.exists('qstates.npy'): # Verifica a existencia do arquivo de Estados salvos.
            Q = np.load('qstates.npy').item() # Carrega o Dicionário previamente salvo.
        else: # Se o arquivo não existir:
            Q = initializeQ() # Cria um novo Dicionário.
    else: # Se quisermos criar um novo Dicionário:
        Q = initializeQ()
    # Rodando os Episódios:
    length = [] # Armazena os números de Ações realizadas em cada Episódio.
    reward = [] # Armazena as Recompensas obtidas em cada Episódio.
    for n in range(N): # Loop que executa N Episódios:
        # Taxa de Exploração:
        explore = explore * EXPLORE_RATE
        # Executando um Episódio:
        episode_reward, episode_length = runEpisode(intervalDist, Q, explore)
        if n % 100 == 0:
            print("Episodio:", n, "| Recompensa:", episode_reward, "| Exploratorion:", explore)
        # Rotina que Salva o Dicionário em sua configuração atual.
        if n % 100 == 0: # A cada 100 Episódios executados o Dicionário é salvo.
            np.save('qstates.npy', Q)
        length.append(episode_length) # Armazena a quantidade de Ações realizadas no Episódio recém executado.
        reward.append(episode_reward) # Armazena a Recompensa Total obtida no Episódio recém executado.
    if PRINT_DICT: # Printa a configuração final do Dicionário após N Episódios executados.
        print(Q)
    print("Estados Visitados:", countStates(Q), "/", MAX_STATES)
    return length, reward

# Função que plota o Gráfico de Aprendizado:
def plotRunningAvg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(running_avg)
    plt.title("Average Reward of Last 100 Episodes")
    plt.ylim([0,MAX_ACTIONS + 10])
    plt.show()

# Função MAIN:
if __name__ == '__main__':
    intervalDist = createIntervalDists()
    # Execução do Treinamento:
    episode_lengths_train, episode_rewards_train = runNEpisodes(intervalDist, EXPLORE_START, 10000)
    plotRunningAvg(episode_rewards_train)
    # Execução do Teste:
    episode_lengths_test, episode_rewards_test = runNEpisodes(intervalDist, 0.0, 500)
    plotRunningAvg(episode_rewards_test)
