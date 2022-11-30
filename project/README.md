# MAC0318 Introdução à Programação de Robôs Móveis

## Projeto final - O Desafio Pato Wheels

Está na hora do **desafio *Pato Wheels***!

Por conta da pandemia e do distanciamento social, essa milenar cerimônia esteve suspensa na
cidade dos patos nos últimos anos. Agora, patos dos mais diferentes cantos e lagos estão
viajando para Patolândia para competir nesse que promete ser o mais histórico
desafio de sempre! Mas temos um problema, o pato-prefeito anterior não terminou as pistas de corrida como prometido e o número de participantes inscritos superou as expectativas. Por isso, o atual pato-prefeito, Paulo Pato, auxiliado pelos seus pato-engenheiros, decidiram utilizar as próprias ruas da cidade como pistas -- com os patos-cidadãos realizando suas atividades rotineiras! 
A competição será disputada em vários circuitos distintos -- em cada circuito vence aquele que terminar uma volta na pista em menor tempo, descontadas as penalizações por infrações. 

Para evitar complicações legais com possíveis incidentes o pato-prefeito e sua equipe técnica decidiram que os patobôs deveriam ser dirigidos de maneira automática por uma inteligência artificial. 
Assim, ninguém pode ser responsabilizado! Um pato-gênio -- so que não!


### Sua missão

Seu objetivo é programar um dos patobôs para terminar o percurso o mais rápido possível respeitando as regras de trânsito e evitando a qualquer custo colocar a vida dos patos-cidadãos em risco.
Os patobôs possuem a configuração usual: duas rodas impulsionadas independentemente e os seguintes sensores:

1. Sua posição e ângulo atual;
2. Imagens instantâneas obtidas por meio da sua câmera frontal; e
3. Se o seu carrinho fugiu para fora da pista ou colidiu contra um objeto.

O Prefeito Paulo Pato comunicou que as três regras da pato-robótica devem ser seguidas:

1. O patobô não deve machucar nenhum pato;
2. O patobô não deve causar danos materiais à cidade ou a outros patobôs;
3. O patobô deve permanecer dentro dos limites da rua; e
4. O patobô deve andar a uma velocidade média igual ou superior a 0.2m/s.

Para cumprir sua missão, vocês podem usar quaisquer técnicas, vistas em aula ou 
não, desde que vocês obedeçam as quatros regras acima e apenas utilizem as informações
dadas. O uso de informações adicionais é proibido.

### Construindo o patobô

O seu código deve ser implementado a partir do arquivo [agent.py](./agent.py), que
contém o código para o agente do Patobô. Como existem imperfeicoes nas ruas da
Patolandia, assume-se um erro nas rodas e portanto temos as constantes usuais para
o cálculo da modelagem `Agent.get_pwm_control`. O método `Agent.send_commands` é 
encarregado de computar as ações do agente. Vocês opcionalmente podem usar, assim
como nas tarefas anteriores, o método `Agent.preprocess` para organizar o seu código
em diferentes módulos de percepção e controle.

**Importante:** todo o código deve ser implementado no arquivo `agent.py`, que será 
o único arquivo código-fonte a ser entregue no e-disciplinas. No entanto, a execução
da tarefa deve ser feita através de [challenge.py](./challenge.py) por meio do comando:

```bash
python3 assignments/challenge/challenge.py assignments/challenge/examples/challenge_n
```

Onde o `n` acima é quaisquer um dos possíveis arquivos `challenge_*` no diretório de
exemplos para execução.

### Verificando seu relatório de infrações

O relatório de infrações é salvo em `/tmp/output.txt` quando o simulador é encerrado 
pela tecla `ESC`. A escala de penalidade não é 1-para-1 com a nota final do projeto. 
Aplicaremos uma constante a cada tipo de penalidade dependendo da dificuldade de
resolução de cada uma delas. Um agente que percorre toda a pista e termina no ponto
inicial sem nenhuma infração a princípio deve receber 10 na nota final (a menos que
informações não permitidas tenham sido utilizadas).

### Entrega

A data maxima de entrega é **21/12/2022**, vocês devem entregar:

1. Seu código `agent.py`;
2. Quaisquer materiais, dados, redes neurais, etc. que são necessários para executar
o código;
3. Um vídeo exibindo o comportamento do agente em algumas pistas.

Em **20/12/2022** e **21/21/2022**, executaremos os diferentes códigos enviados ao final da aula, para ilustrar as diferentes abordagens e comportamentos obtidos. 

### Bônus

Além das infrações, o relatório do Patobô também contém uma medição do tempo de
execucao usado pelo competidor para finalizar a pista. O time de Patobô que for
mais rapido em cada desafio um bônus na nota do projeto!

### Comunicação entre grupos

Os grupos não devem compartilhar informações de suas soluções. Porém, é permitido
compartilhar os vídeos de execução dos seus agentes entre grupos. Para gravar videos,
vocês podem utilizar a gravação automática do simulador por meio da variável de
construção `video_path` na criação do ambiente em [challenge.py](./challenge.py).

**Boa sorte e bom trabalho! Patolândia conta com vocês!**
