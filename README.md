## Resumo

Essa API foi desenvolvido como trabalho de formatura do grupo Felipe Malberiger, Nazli Setton e Rodrigo Sabino da Escola Politécnica da USP para o curso de Engenharia de Computação. O projeto se resume em uma API que recebe um arquivo de áudio e através de algoritmos de machine learning o classifica entre narração ou música.


## Acesso e Dados de Entrada da API

Para acessar a API é necessário fazer uma requisição POST no endereço http://localhost:5000/speechmusic/[algoritmo] contendo o áudio em anexo. A frequência do áudio deve ser de 44100Hz. 

O [algoritmo] no endereço define qual classificador será utilizado na predição. As opções são: 

- knn (K-Nearest Neighbors)
- knne (K-Nearest Neighbors com apenas características de energia)
- nn (Neural Network)
- nne (Neural Network com apenas características de energia)
- svm (Support Vector Machine)
- svme (Support Vector Machine com apenas características de energia)
- voter (Todos os algoritmos são executados com um votador que consolida os resultados)
- votere (Todos os algoritmos com apenas características de energia são executados com um votador que consolida os resultados)

## Funcionalidades de Cada Arquivo

**application.py** - API criada em Flask

**predict_audio.py** - Classificação do áudio

**variables_extraction.py** - Extração das características do áudio

**feature_extraction.py** - Cálculo dos atributos do áudio

**\*.json** - Arquivos das configurações e variáveis utilizadas no projeto

## Instalação do YAAFE

A instalação da biblioteca YAAFE é feita através do gerenciado de arquivos Conda (https://conda.io/docs/) e pode ser feito com os seguintes comandos:

1) wget http://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
 
2) bash Anaconda3-4.3.1-Linux-x86_64.sh
3) source ~/.bashrc
4) conda install -c conda-forge yaafe=0.70

## Retorno da API

Após processar o áudio a API retorna um arquivo JSON no formato:

{'audio_file': [nomearquivo],
[algoritmo1]: {
                'sec_by_sec_music_prediction': [arrayComPrevisaoDeCadaSegundo],
              	'likely_to_be_music': [probabilidadeDeSerMusica]
              },
[algoritmo2]: {...}
'voter': {
			'sec_by_sec_music_prediction': [arrayComPrevisaoDeCadaSegundo],
         	'likely_to_be_music': [probabilidadeDeSerMusica]
         }

Do arquivo JSON acima, temos:

– [nomearquivo]: Nome do áudio enviado.

– [algoritmo1/2]: Abreviação do algoritmo usado na classificação.

– [arrayComPrevisaoDeCadaSegundo]: Uma lista cujo número de elementos é igualao tempo do áudio em segundos.  Cada elemento pode ser zero ou um, assim, caso seja zero, aquele segundo foi previsto como fala e caso seja um, foi previsto comomúsica.

– [ProbabilidadeDeSerMusica]:  Probabilidade do áudio todo ser música.  Este parâmetro é calculado somando todos os elementos do [arrayComPrevisaoDeCadaSegundo] e dividindo pelo seu tamanho.

– Os valores com as chaves voter e algoritmo2 apenas aparecem quando o algoritmo escolhido é voter ou votere.
