import streamlit as st
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from eda import column_description, heart_df, valores, heart_df_new, a, fighist, figbox
from rmvnoise import heart_df_no_outlier, remove_outliers_code, valoresnout, fignout, figboxnout, fighistnout


class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def subheader2(self, text):
        self._markdown(text, "h4", " " * 6)

    def subheader3(self, text):
        self._markdown(text, "h5", " " * 8)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):

        check_text = lambda x: x if x.isalnum() else '-'
        key = "".join([check_text(x) for x in text]).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


toc = Toc()
st.set_page_config(page_title="Heart Failure Dashboard", layout="centered",
                   page_icon='https://twemoji.maxcdn.com/v/13.1.0/72x72/1fac0.png')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.sidebar.title("Table of Contents")
toc.placeholder(sidebar=True)
toc.title("Heart Failure Dashboard - A Machine Learning Project")


check_box_all = st.sidebar.checkbox(label="Show All")

if check_box_all:
    val = True
else:
    val = False

check_box_col = st.sidebar.checkbox(label="Show Columns description", value=val)
check_box_eda = st.sidebar.checkbox(label="Show Exploratory Data Analysis", value=val)
check_box_noise = st.sidebar.checkbox(label="Show Removing noise from data", value=val)
check_box_tv = st.sidebar.checkbox(label="Show Training and Validation", value=val)
check_box_bn = st.sidebar.checkbox(label="Show Bonus Section", value=val)

if check_box_col:
    toc.header('Dataset description')
    st.markdown('---')
    st.markdown(column_description)
    st.write('\n')

if check_box_eda:

    toc.header("Exploratory Data Analysis")
    st.markdown("---")

    toc.subheader("Loading the dataset")
    st.write(
        'Inicialmente, é preciso carregar os dados presentes no arquivo heart_failure_clinical_records_dataset.csv. '
        'Para isso, utilizaremos a o método read_csv da biblioteca Pandas')
    st.code('DATA_PATH = "Dados/heart_failure_clinical_records_dataset.csv"\n'
            'heart_df = pd.read_csv(DATA_PATH)\n'
            'heart_df.head()')
    st.write(heart_df.head())

    toc.subheader('Looking at data shape')
    st.write('Observando o shape do dataset verifica-se que ele possui 299 instâncias e 13 atributos, sendo um dos '
             'atributos a nossa variável target')
    st.code('heart_df.shape()')
    st.write(heart_df.shape)

    toc.subheader('Checking features data types')
    st.write('Todas as features já são de tipos númericos, sendo assim, não há a necessidade de converter dados '
             'categóricos para númericos')
    st.code('heart_df.dtypes')
    table = '| Feature                  | dtype      |\n' \
            '|--------------------------|------------|'
    for dtype, column in zip(heart_df.dtypes.to_list(), heart_df.columns.to_list()):
        hifnum_c = 26 - len(column)
        hif_c = ' '*hifnum_c
        hifnum_d = 26 - len(str(dtype))
        hif_d = ' '*hifnum_d
        table = table + f'\n|{column}{hif_c}|{str(dtype)}{hif_d}|'
    st.markdown(table)
    st.write('\n')

    toc.subheader('Statistical description of the dataset')
    st.write('Conjunto de medidas estatísticas associadas a cada atributo, apenas observando esses dados pode-se '
             'perceber que os dados possuem variâncias (std^2) em escalas bem diferentes, sendo necessário lidar com '
             'isso nas etapas de processamento dos dados, antes das aplicações dos modelos de machine learning.')
    st.code('heart_df.describe()')
    st.write(heart_df.describe())

    toc.subheader('Checking for missing values')
    st.write("O dataset não possui nenhum valor faltante, sendo assim não precisaremos aplicar nenhum tipo de imputação"
             "para esse tipo de ruído")
    st.code('msno.bar(heart_df)')
    fignull = plt.figure(figsize=(10,8))
    ax = fignull.add_subplot(1,1,1)
    bar = msno.bar(heart_df, ax=ax)
    st.pyplot(fignull, transparent=True, dpi=300)

    toc.subheader('Looking at the target variable (DEATH_EVENT)')
    st.write('Como se pode observar, os dados da variável de saída não possuem a mesma proporção, existem cerca de 2 '
             'vezes mais pacientes que não morreram durante o período em que foram acompanhados. Sendo assim, será '
             'necessário lidar com esses dados desbalanceados na etapa de processamento.')
    st.write(heart_df.DEATH_EVENT.value_counts())
    toc.subheader2('Count plot of the target variable')
    st.write('Não faleceram: {:.2f}% dos casos ({:.0f})'.format(valores[0]/(valores[1]+valores[0])*100, valores[0]))
    st.write('Faleceram: {:.2f}% dos casos ({:.0f})'.format(valores[1]/(valores[1]+valores[0])*100, valores[1]))
    st.write('Proporção dos dados de saída: {:.2f}'.format(valores[0]/valores[1]))
    figtarget = plt.figure()
    ax1 = figtarget.add_subplot(1,1,1)
    count_target = sns.countplot(data=heart_df, x='DEATH_EVENT', ax=ax1)
    st.pyplot(figtarget, transparent=True)

    toc.subheader('Correlation between variables')
    st.write('Observando o mapa de calor das correlações pode-se perceber que:')
    figcorr = plt.figure(figsize=(10,8))
    ax2 = figcorr.add_subplot(1,1,1)
    count_target = sns.heatmap(heart_df.corr(), annot=True, cmap='viridis', fmt='.3f', ax=ax2)
    st.pyplot(figcorr, transparent=True)
    st.markdown("* Existe uma correlação moderada entre a idade e se o paciente morreu, ou seja, quanto maior a idade, "
                "maiores as chances do paciente ter morrido.\n"
                "* Existe uma correlação negativa moderada entre a fração de ejeção e se o paciente morreu, ou seja, "
                "quanto menor a porcentagem de sangue saindo do coração a cada contração, maiores as chances do paciente"
                " ter morrido.\n"
                "* Existe uma correlação moderada entre a Creatinina sérica e se o paciente morreu, ou seja, quanto "
                "maior o nível de creatinina no sangue, maiores as chances do paciente ter morrido.\n"
                "* Existe uma correlação negativa leve para moderada entre o Sódio sérico e se o paciente morreu, ou "
                "seja, quanto menor o nível de sódio no sangue, maiores as chances do paciente ter morrido.\n"
                "* Existe uma correlação negativa moderada para forte entre o tempo de acompanhamento e se o paciente "
                "morreu, ou seja, quanto menor o período de acompanhamento do paciente, maiores as chances do paciente "
                "ter morrido.\n")
    toc.subheader2('Dropping features with less than 1% of correlation with our target')
    st.write('Foram removidos os atributos \'diabetes\' e \'sex\'')
    st.code('bigger_than_1perc = np.abs(heart_df.corr()[\'DEATH_EVENT\']) > 0.01\n'
            'new_features_list = heart_df.corr()[bigger_than_1perc][\'DEATH_EVENT\'].index.to_list()\n'
            'heart_df_new = heart_df[new_features_list]\n'
            'heart_df_new.columns')
    st.write(heart_df_new.columns)

    toc.subheader('Checking the mean values for the features relative to the DEATH_EVENT variable')
    st.write('Observando um pouco é possível perceber que existem pequenas diferenças entre os dois grupos')
    st.code('heart_df_new.groupby([\'DEATH_EVENT\']).mean()')
    st.write(heart_df_new.groupby(['DEATH_EVENT']).mean())

    toc.subheader('Looking at variance for continuous features')
    st.write('Os valores das variâncias dos atributos contínuos estão em escalas bem diferentes, sendo assim, será'
             'necessário fazer o tratamento desses dados para alguns dos modelos de machine learning que são sensíveis'
             'às variâncias em escalas diferentes')
    st.write(a)

    toc.subheader('Looking at features Histograms')
    st.pyplot(fighist, transparent=True)

    toc.subheader('Looking at Boxplots for continuous features')
    st.write('Existem alguns outliers que precisam ser removidos para não enviesar nossos modelos de classificação')
    st.pyplot(figbox, transparent=True)

if check_box_noise:
    toc.header('Removing noise from data')
    st.markdown('---')
    toc.subheader('Removing outliers from continuous features')
    st.write('Para a remoção dos outliers eu defini uma função chamada \'remove_outliers\' que utiliza o método de '
             'remoção baseado no score Z, ou seja, se o valor for maior que a média + 3 * desvios padrão, ou for menor'
             'que a média - 3 * desvios padrão, ele será considerado um outlier e será removido dos nossos dados')
    remov = st.checkbox(label='Show remove_outliers code')
    if remov:
        st.code(remove_outliers_code)
    st.code('heart_df_no_outlier,outliers_ind = remove_outliers(heart_df_new, columns_continuous, outliers_index=True)')
    st.write('Observando os shapes dos datasets vemos que foram descartadas 19 instâncias')
    st.write(f'Instâncias heart_df_new: {heart_df_new.shape[0]}')
    st.write(f'Instâncias heart_df_no_outlier: {heart_df_no_outlier.shape[0]}')
    toc.subheader2('Count plot of the target variable without outliers')
    st.write('Não faleceram: {:.2f}% dos casos'.format(valoresnout[0]/(valoresnout[1]+valoresnout[0])*100))
    st.write('Faleceram: {:.2f}% dos casos'.format(valoresnout[1]/(valoresnout[1]+valoresnout[0])*100))
    st.write('Proporção dos dados de saída: {:.2f}'.format(valoresnout[0]/valoresnout[1]))
    st.pyplot(fignout, transparent=True)

    toc.subheader2('Looking at Boxplots without outliers for continuous features')
    st.write('Observando os boxplots é visível a diferença, foi possível remover com sucesso os outliers, sobrando '
             'apenas os valores dentro do intervalo média +/- 3*DP (É válido lembrar que os boxplots apresentam alguns'
             ' outliers pois usam uma métrica um pouco diferente para classificá-los')
    st.pyplot(figboxnout, transparent=True)

    toc.subheader2('Looking at Probability Distribution for continuous features')
    st.pyplot(fighistnout, transparent=True)



#
# if check_box_tv:
#
# if check_box_bn:

toc.generate()


#
# st.markdown("---")
# st.markdown("## Descrição")
# if check_box_col:
#     st.markdown(markdown_1)
#
# st.markdown("---")
# st.markdown("## Visualização dos Dados")
#
# st.markdown(markdown_2)
# st.plotly_chart(fig1, use_container_width=True)
# st.markdown(markdown_3)
# st.plotly_chart(fig2, use_container_width=True)
# st.markdown(markdown_4)
# st.plotly_chart(fig3, use_container_width=True)
# st.markdown(markdown_5)
# st.plotly_chart(fig4, use_container_width=True)
# st.plotly_chart(fig5, use_container_width=True)
# st.markdown(markdown_7)
# st.plotly_chart(fig6, use_container_width=True)
# st.markdown(markdown_6)
# st.plotly_chart(fig7, use_container_width=True)
# st.plotly_chart(fig8, use_container_width=True)
#
# st.markdown(markdown_8, unsafe_allow_html=True)

