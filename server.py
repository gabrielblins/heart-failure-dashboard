import pandas as pd
import streamlit as st
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from eda import column_description, heart_df, valores, heart_df_new, a, fighist, figbox
from rmvnoise import heart_df_no_outlier, remove_outliers_code, valoresnout, fignout, figboxnout, fighistnout
from trainandval import report_tree, tree_fig, figtreecm, figroc, sort_features_tree, report_rf, figrfcm, figrocrf, \
                        sort_features_rf, report_rf_time, report_rf_disc, report_rf_disc_time
from bonus import objective_rf_code, study_rf, report_rf_best, figrfbestcm, figrocrfbest, pruned, complete, val_best, \
                  trial_rf, optimization_hist_rf, parallel_coordinate_rf, sort_features_rf_best, report_svm, \
                  report_svm_time, report_xgb, report_xgb_best, objective_code, finished, Value, trial, \
                  optimization_hist_xgb, parallel_coordinate_xgb, figxgbbestcm, figrocxgbbest, sort_features_xgb_best


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
check_box_conc = st.sidebar.checkbox(label="Show Conclusion", value=val)

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
        'Inicialmente, ?? preciso carregar os dados presentes no arquivo heart_failure_clinical_records_dataset.csv. '
        'Para isso, utilizaremos a o m??todo read_csv da biblioteca Pandas')
    st.code('DATA_PATH = "Dados/heart_failure_clinical_records_dataset.csv"\n'
            'heart_df = pd.read_csv(DATA_PATH)\n'
            'heart_df.head()')
    st.write(heart_df.head())

    toc.subheader('Looking at data shape')
    st.write('Observando o shape do dataset verifica-se que ele possui 299 inst??ncias e 13 atributos, sendo um dos '
             'atributos a nossa vari??vel target')
    st.code('heart_df.shape()')
    st.write(heart_df.shape)

    toc.subheader('Checking features data types')
    st.write('Todas as features j?? s??o de tipos n??mericos, sendo assim, n??o h?? a necessidade de converter dados '
             'categ??ricos para n??mericos')
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
    st.write('Conjunto de medidas estat??sticas associadas a cada atributo, apenas observando esses dados pode-se '
             'perceber que os dados possuem vari??ncias (std^2) em escalas bem diferentes, sendo necess??rio lidar com '
             'isso nas etapas de processamento dos dados, antes das aplica????es dos modelos de machine learning.')
    st.code('heart_df.describe()')
    st.write(heart_df.describe())

    toc.subheader('Checking for missing values')
    st.write("O dataset n??o possui nenhum valor faltante, sendo assim n??o precisaremos aplicar nenhum tipo de imputa????o"
             "para esse tipo de ru??do")
    st.code('msno.bar(heart_df)')
    fignull = plt.figure(figsize=(10,8))
    ax = fignull.add_subplot(1,1,1)
    bar = msno.bar(heart_df, ax=ax)
    st.pyplot(fignull, transparent=True, dpi=300)

    toc.subheader('Looking at the target variable (DEATH_EVENT)')
    st.write('Como se pode observar, os dados da vari??vel de sa??da n??o possuem a mesma propor????o, existem cerca de 2 '
             'vezes mais pacientes que n??o morreram durante o per??odo em que foram acompanhados. Sendo assim, ser?? '
             'necess??rio lidar com esses dados desbalanceados na etapa de processamento.')
    st.write(heart_df.DEATH_EVENT.value_counts())
    toc.subheader2('Count plot of the target variable')
    st.write('N??o faleceram: {:.2f}% dos casos ({:.0f})'.format(valores[0]/(valores[1]+valores[0])*100, valores[0]))
    st.write('Faleceram: {:.2f}% dos casos ({:.0f})'.format(valores[1]/(valores[1]+valores[0])*100, valores[1]))
    st.write('Propor????o dos dados de sa??da: {:.2f}'.format(valores[0]/valores[1]))
    figtarget = plt.figure()
    ax1 = figtarget.add_subplot(1,1,1)
    count_target = sns.countplot(data=heart_df, x='DEATH_EVENT', ax=ax1)
    st.pyplot(figtarget, transparent=True)

    toc.subheader('Correlation between variables')
    st.write('Observando o mapa de calor das correla????es pode-se perceber que:')
    figcorr = plt.figure(figsize=(10,8))
    ax2 = figcorr.add_subplot(1,1,1)
    count_target = sns.heatmap(heart_df.corr(), annot=True, cmap='viridis', fmt='.3f', ax=ax2)
    st.pyplot(figcorr, transparent=True)
    st.markdown("* Existe uma correla????o moderada entre a idade e se o paciente morreu, ou seja, quanto maior a idade, "
                "maiores as chances do paciente ter morrido.\n"
                "* Existe uma correla????o negativa moderada entre a fra????o de eje????o e se o paciente morreu, ou seja, "
                "quanto menor a porcentagem de sangue saindo do cora????o a cada contra????o, maiores as chances do paciente"
                " ter morrido.\n"
                "* Existe uma correla????o moderada entre a Creatinina s??rica e se o paciente morreu, ou seja, quanto "
                "maior o n??vel de creatinina no sangue, maiores as chances do paciente ter morrido.\n"
                "* Existe uma correla????o negativa leve para moderada entre o S??dio s??rico e se o paciente morreu, ou "
                "seja, quanto menor o n??vel de s??dio no sangue, maiores as chances do paciente ter morrido.\n"
                "* Existe uma correla????o negativa moderada para forte entre o tempo de acompanhamento e se o paciente "
                "morreu, ou seja, quanto menor o per??odo de acompanhamento do paciente, maiores as chances do paciente "
                "ter morrido.\n")
    toc.subheader2('Dropping features with less than 1% of correlation with our target')
    st.write('Foram removidos os atributos \'diabetes\' e \'sex\'')
    st.code('bigger_than_1perc = np.abs(heart_df.corr()[\'DEATH_EVENT\']) > 0.01\n'
            'new_features_list = heart_df.corr()[bigger_than_1perc][\'DEATH_EVENT\'].index.to_list()\n'
            'heart_df_new = heart_df[new_features_list]\n'
            'heart_df_new.columns')
    st.write(heart_df_new.columns)

    toc.subheader('Checking the mean values for the features relative to the DEATH_EVENT variable')
    st.write('Observando um pouco ?? poss??vel perceber que existem pequenas diferen??as entre os dois grupos')
    st.code('heart_df_new.groupby([\'DEATH_EVENT\']).mean()')
    st.write(heart_df_new.groupby(['DEATH_EVENT']).mean())

    toc.subheader('Looking at variance for continuous features')
    st.write('Os valores das vari??ncias dos atributos cont??nuos est??o em escalas bem diferentes, sendo assim, ser??'
             'necess??rio fazer o tratamento desses dados para alguns dos modelos de machine learning que s??o sens??veis'
             '??s vari??ncias em escalas diferentes')
    st.write(a)

    toc.subheader('Looking at features Histograms')
    st.pyplot(fighist, transparent=True)

    toc.subheader('Looking at Boxplots for continuous features')
    st.write('Existem alguns outliers que precisam ser removidos para n??o enviesar nossos modelos de classifica????o')
    st.pyplot(figbox, transparent=True)

if check_box_noise:
    toc.header('Removing noise from data')
    st.markdown('---')
    toc.subheader('Removing outliers from continuous features')
    st.write('Para a remo????o dos outliers eu defini uma fun????o chamada \'remove_outliers\' que utiliza o m??todo de '
             'remo????o baseado no score Z, ou seja, se o valor for maior que a m??dia + 3 * desvios padr??o, ou for menor'
             'que a m??dia - 3 * desvios padr??o, ele ser?? considerado um outlier e ser?? removido dos nossos dados')
    remov = st.checkbox(label='Show remove_outliers code')
    if remov:
        st.code(remove_outliers_code)
    st.code('heart_df_no_outlier,outliers_ind = remove_outliers(heart_df_new, columns_continuous, outliers_index=True)')
    st.write('Observando os shapes dos datasets vemos que foram descartadas 19 inst??ncias')
    st.write(f'Inst??ncias heart_df_new: {heart_df_new.shape[0]}')
    st.write(f'Inst??ncias heart_df_no_outlier: {heart_df_no_outlier.shape[0]}')
    toc.subheader2('Count plot of the target variable without outliers')
    st.write('N??o faleceram: {:.2f}% dos casos'.format(valoresnout[0]/(valoresnout[1]+valoresnout[0])*100))
    st.write('Faleceram: {:.2f}% dos casos'.format(valoresnout[1]/(valoresnout[1]+valoresnout[0])*100))
    st.write('Propor????o dos dados de sa??da: {:.2f}'.format(valoresnout[0]/valoresnout[1]))
    st.pyplot(fignout, transparent=True)

    toc.subheader2('Looking at Boxplots without outliers for continuous features')
    st.write('Observando os boxplots ?? vis??vel a diferen??a, foi poss??vel remover com sucesso os outliers, sobrando '
             'apenas os valores dentro do intervalo m??dia +/- 3*DP (?? v??lido lembrar que os boxplots apresentam alguns'
             ' outliers pois usam uma m??trica um pouco diferente para classific??-los')
    st.pyplot(figboxnout, transparent=True)

    toc.subheader2('Looking at Probability Distribution for continuous features')
    st.pyplot(fighistnout, transparent=True)

if check_box_tv:
    toc.header('Training and Validation')
    st.markdown('---')

    st.subheader('Choosing our features and setting our target')
    st.write('Agora ?? preciso dividir nossos dados em entrada(X) e sa??da(y), nesse caso o X s??o todas as features'
             'exceto os atributos \'time\' e \'DEATH_EVENT\', para ser poss??vel avaliar nosso modelo apenas com dados '
             'relacionados as condi????es fisiologicas e estilo de vida do paciente. J?? o X_t cont??m o tempo de '
             'acompanhamento para que seja poss??vel analisar o quanto esse atributo influencia no score dos modelos')
    st.code('X = heart_df_no_outlier.drop([\'DEATH_EVENT\',\'time\'], axis=1)\n'
            'X_t = heart_df_no_outlier.drop(\'DEATH_EVENT\', axis=1)\n'
            'y = heart_df_no_outlier[\'DEATH_EVENT\']')

    st.subheader('Splitting our data into training and test')
    st.write('Os dados foram dividos em treino e teste usando a fun????o train_test_split da biblioteca Scikit-Learn, um '
             'grupo sem o atributo \'time\' e o outro com esse atributo.')
    st.write('Foi utilizado a divis??o 70% para treino e 30% para teste, al??m disso, foi utilizado o par??metro '
             'stratify=y para que se mantenha a propor????o da vari??vel de sa??da nos dados de treino e de teste.')
    st.code('X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=5, stratify=y)\n'
            'X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t,y,test_size=.3,random_state=5,stratify=y)')

    toc.subheader('Data Pre-Processing')
    st.write('Nessa etapa s??o gerados mais 4 tipos de dados de treino e teste a partir dos dados que j?? foram gerados'
             ', 2 deles para os dados sem o atributo \'time\' e 2 para os dados com esse atributo. (Sempre importante '
             'lembrar que esses transformadores s?? devem ser ajustados (.fit) nos dados de treino, para n??o ter nenhum '
             'vi??s dos dados de teste')
    toc.subheader2('Scaling the data')
    st.write('Aqui ocorre a normaliza????o dos dados de treino e teste usando a fun????o MinMaxScaler da biblioteca '
             'Scikit-Learn, gerando dois novos conjunto de dados, um sem o atributo \'time\' e outro com esse atributo.'
             ' Isso ?? necess??rio para que todas as features estejam na mesma escala, solucionando o problema da grande '
             'diferen??a de vari??ncia entre os atributos')
    st.code('scaler = MinMaxScaler()')
    st.code('# Scaled without Time\n'
            'X_train_s = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns.to_list())\n'
            'X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns.to_list())\n'
            '# Scaled with Time\n'
            'X_train_st = pd.DataFrame(scaler.fit_transform(X_train_t), columns=X_train_t.columns.to_list())\n'
            'X_test_st = pd.DataFrame(scaler.transform(X_test_t), columns=X_test_t.columns.to_list())')
    toc.subheader2('Discretizing the data')
    st.write('Aqui s??o gerados mais dois conjuntos de dados, dessa vez aplicando uma t??cnica de discretiza????o das '
             'vari??veis cont??nuas usando a fun????o EqualFrequencyDiscretiser da biblioteca Feature Engine, pode ser que '
             'isso traga melhores resultados para os nossos modelos')
    st.code('discretizer = EqualFrequencyDiscretiser(variables=columns_continuous_not, q=8)\n'
            'discretizer_t = EqualFrequencyDiscretiser(variables=columns_continuous, q=8)')
    ch_disc = st.checkbox(label='Show fit and transform for discretizer')
    if ch_disc:
        st.code(
"""
#Discrete without Time
discrete = discretizer.fit_transform(X_train)
X_train_d = pd.DataFrame()
for column in X_train.columns.to_list():
    if column in discrete.columns.to_list():
        X_train_d[column] = discrete[column]
    else:
        X_train_d[column] = X_train[column]
discrete_test = discretizer.transform(X_test)
X_test_d = pd.DataFrame()
for column in X_test.columns.to_list():
    if column in discrete_test.columns.to_list():
        X_test_d[column] = discrete_test[column]
    else:
        X_test_d[column] = X_test[column]

# Discrete with Time
discrete_t = discretizer_t.fit_transform(X_train_t)
X_train_dt = pd.DataFrame()
for column in X_train_t.columns.to_list():
    if column in discrete_t.columns.to_list():
        X_train_dt[column] = discrete_t[column]
    else:
        X_train_dt[column] = X_train_t[column]
discrete_test_t = discretizer_t.transform(X_test_t)
X_test_dt = pd.DataFrame()
for column in X_test_t.columns.to_list():
    if column in discrete_test_t.columns.to_list():
        X_test_dt[column] = discrete_test_t[column]
    else:
        X_test_dt[column] = X_test_t[column]
"""
                )

    toc.subheader2('Applying SMOTE to handle with imbalanced data')
    st.write('Para lidar com o problema de desbalanceamento da vari??vel de sa??da foi utilizado o m??todo de oversampling'
             'SMOTE (Synthetic Minority Over-sampling Technique) para gerar novas inst??ncias sint??ticas para a classe '
             'minorit??ria, esse m??todo funciona usando a t??cnica de K vizinhos mais pr??ximos para selecionar exemplos '
             'que estejam pr??ximos no espa??o de atributos, desenhando uma linha entre os exemplos e, a partir disso, '
             'gera uma nova inst??ncia da classe minorit??ria em algum ponto dessa linha.')
    st.write('Aqui s??o usadas duas fun????es da biblioteca Imbalanced-Learn, criadas a partir do SMOTE, a SMOTENC'
             ', que funciona com os atributos cont??nuos e categ??ricos, e a SMOTEN que deve ser usada apenas para dados'
             'categ??ricos. (Importante lembrar que o oversampling s?? pode ser aplicado nos dados de treino)')
    st.markdown('* Para os conjuntos onde n??o foi aplicado a discretiza????o foi usada a fun????o SMOTENC. Foi passado o '
                'par??metro categorical_features com a posi????o das features categ??ricas presentes.')
    st.code('smote = SMOTENC(categorical_features= [1, 4, 8],random_state=5)\n'
            '# Sem scaling\n'
            'X_train_res, y_train_res = smote.fit_resample(X_train,y_train)\n'
            'X_train_res_t, y_train_res_t = smote.fit_resample(X_train_t,y_train_t)\n'
            '# Com scaling\n'
            'X_train_s_res, y_train_s = smote.fit_resample(X_train_s,y_train)\n'
            'X_train_st_res, y_train_st = smote.fit_resample(X_train_st,y_train_t)')
    st.markdown('* Para os conjuntos de treino onde foi aplicado a discretiza????o foi usada a fun????o SMOTEN')
    st.code('X_train_d_res, y_train_d = smotec.fit_resample(X_train_d,y_train)\n'
            'X_train_dt_res, y_train_dt = smotec.fit_resample(X_train_dt,y_train_t)')

    toc.subheader('Choosing the best Tree based model')
    st.write('Nessa etapa, ?? feito um estudo com os modelos baseados em ??rvore mais comuns, as Decision Trees e as '
             'Random Forests')
    toc.subheader2('Using Decision Tree to make the initial classification')
    st.write('As ??rvores de decis??o s??o modelos de aprendizado supervisionado que podem ser usados tanto para '
             'classifica????o como para regress??o. As Decision Trees buscam gerar um modelo que preveja a vari??vel target'
             ', aprendendo regras de decis??o simples inferidas das features do conjunto de dados. ?? um modelo que pode '
             'ser chamado de "Caixa Branca" devido por gerarem resultados intuitivos e de f??cil interpreta????o.')
    st.write('Ser?? utilizado o modelo DecisionTreeClassifier da biblioteca Scikit-Learn. Foi utilizado o par??metro '
             'max_depth = 5 para mais f??cil visualiza????o dos resultados.')
    st.code('clf_tree = DecisionTreeClassifier(random_state=5, max_depth=5)\n'
            'clf_tree.fit(X_train_res,y_train_res)')
    st.write('Ap??s ajustar o modelo clf_tree aos dados de treino pode-se avaliar o modelo usando os dados de teste. '
             'Para avaliar o modelo est?? sendo utilizada a fun????o classification_report da biblioteca Scikit-Learn, que'
             'apresenta v??rias m??tricas ??teis para avaliar modelos de classifica????o.')
    st.code('y_pred = clf_tree.predict(X_test)\n'
            'classification_report(y_test,y_pred)')
    st.table(pd.DataFrame(report_tree).T)
    st.write('Pode-se observar que o modelo clf_tree est?? com cerca de 74% de acur??cia, por??m essa m??trica n??o ?? a '
             'melhor para avaliar conjunto de dados desbalanceados, para uma maior certeza da validade do modelo ?? '
             'poss??vel observar o F1 Score, definido pela m??dia harm??nica entre precis??o e recall, onde a precis??o vai '
             'dizer o qu??o bem o modelo est?? acertando a classe positiva (nesse caso quando o paciente morre), j?? o '
             'recall(sensibilidade) vai indicar que porcentagem dos valores previstos como positivo ?? realmente '
             'positivo, ?? uma m??trica muito ??til em casos onde os falsos negativos(positivos que foram classificados '
             'como negativos) s??o importantes para a an??lise. Nesse estudo de caso, o recall ?? de extrema import??ncia, '
             'pois se um paciente tiver uma alta chance de vir a ??bito por insufici??ncia card??aca e for classificado '
             'como \'N??o vai falecer\' ser?? um grande problema, j?? que ele n??o receber?? o tratamento adequado, '
             'aumentando mais ainda as chances de ??bito.')
    toc.subheader3('Plotting the Decision Tree choices')
    st.pyplot(tree_fig, transparent=True)
    toc.subheader3('Confusion Matrix for decision tree')
    st.pyplot(figtreecm, transparent=True)
    toc.subheader3('ROC curve for Decision Tree')
    st.write('A ??rea abaixo da curva (AUC) ROC ?? outra boa m??trica para avaliar modelos desbalanceados, j?? que ela n??o '
             'depende da distribui????o da vari??vel de sa??da, a curva ROC mostra o trade-off entre o True Positive Rate '
             '(sensibilidade) e o False Positive Rate (1 - especificidade), sendo assim, quanto mais '
             'pr??ximo a curva estiver do canto superior esquerdo, melhor ?? o nosso modelo. Usa-se a AUC como uma forma'
             ' de resumir a ROC em uma ??nica m??trica. No caso do modelo usando ??rvore de Decis??o foi obtido um AUC de'
             '0.7, ou seja para cerca de 70% dos casos de teste o modelo consegue distinguir corretamente entre '
             'pacientes que n??o faleceram (0) e que faleceram(1)')
    st.pyplot(figroc, transparent=True)
    toc.subheader3('Feature importances for the decision tree classifier')
    st.write('Al??m das m??tricas, ?? poss??vel observar quais atributos est??o sendo mais importantes para o modelo efetuar'
             ' a classifica????o.')
    st.write('Ordem de import??ncia dos atributos usando a Decision Tree:')
    for key,value in zip(sort_features_tree.keys(), sort_features_tree.values()):
        st.write(f'{key}:',value)

    toc.subheader2('Using Random Forest to improve the score')
    st.write('As Random Forests s??o um conjunto de ??rvores de decis??o, cada ??rvore nesse conjunto ?? ajustada com uma '
             'amostra diferente dos dados de treino com reposi????o usando usando Bootstrap como m??todo de reamostragem.'
             ' Para determinar a classe de sa??da as Random Forests calculam a m??dia das probabilidades e escolhem a '
             'classe que possuir a maior m??dia. As Random Forests normalmente se saem melhor em rela????o as ??rvores de '
             'decis??o, pois a combina????o de v??rias ??rvores faz com que haja uma redu????o significativa na vari??ncia, a '
             'custo de um pequeno aumento no vi??s do modelo, o que leva a uma melhor generaliza????o dos dados de treino,'
             ' o que resulta num modelo melhor.')
    st.code('clf_rf = RandomForestClassifier(random_state=5)\n'
            'clf_rf.fit(X_train_res,y_train_res)')
    st.write('Ap??s ajustar o modelo clf_rf aos dados de treino pode-se avaliar o modelo usando os dados de teste. Mais'
             'uma vez foi usada as m??tricas providas pela fun????o classification_report do Scikit')
    st.code('y_pred = clf_rf.predict(X_test)\n'
            'classification_report(y_test,y_pred)')
    st.table(pd.DataFrame(report_rf).T)
    st.write('Apesar do recall ter permanecido o mesmo, houve um aumento consider??vel na precis??o, consequentemente, '
             'o modelo clf_rf obteve um melhor F1 Score em rela????o ao modelo anterior usando ??rvore de decis??o e tamb??m'
             'obteve uma melhor acur??cia.')
    toc.subheader3('Confusion Matrix for Random Forest')
    st.pyplot(figrfcm, transparent=True)
    toc.subheader3('ROC curve for Random Forest')
    st.write('Como esperado pelas m??tricas anteriores, o valor de AUC para o modelo usando Random Forest aumentou em 3%'
             ', sendo um modelo mais interessante para fazer ajustes de hiperpar??metros e obter melhores resultados.')
    st.pyplot(figrocrf, transparent=True)
    toc.subheader3('Feature importances for the Random Forest classifier')
    st.write('Ordem de import??ncia dos atributos usando a Random Forest:')
    for key,value in zip(sort_features_rf.keys(), sort_features_rf.values()):
        st.write(f'{key}:',value)

    toc.subheader('Trying to improve the scores using the other train/test sets')
    st.write('Agora que o modelo base j?? foi escolhido, vale a pena testar os outros conjuntos de treino.')
    st.markdown('* Usando o atributo \'time\'')
    st.code('clf_rf.fit(X_train_res_t,y_train_res_t)\n'
            'y_pred = clf_rf.predict(X_test_t)\n'
            'classification_report(y_test_t,y_pred)')
    st.table(pd.DataFrame(report_rf_time).T)
    st.markdown('* Sem o atributo \'time\', Discretizado')
    st.code('clf_rf.fit(X_train_d_res,y_train_d)\n'
            'y_pred = clf_rf.predict(X_test_d)\n'
            'classification_report(y_test,y_pred)')
    st.table(pd.DataFrame(report_rf_disc).T)
    st.markdown('* Usando o atributo \'time\', Discretizado')
    st.code('clf_rf.fit(X_train_dt_res,y_train_dt)\n'
            'y_pred = clf_rf.predict(X_test_dt)\n'
            'classification_report(y_test_t,y_pred)')
    st.table(pd.DataFrame(report_rf_disc_time).T)
    st.write('Observando os resultados obtidos, ?? poss??vel perceber que o modelo fica melhor usando o atributo \'time\''
             ' e sem discretizar os atributos cont??nuos. Sendo assim, para as pr??ximas etapas ser?? usado o conjunto de '
             'dados contendo o atributo \'time\'')

if check_box_bn:
    toc.header('Bonus Section')
    st.markdown('---')
    toc.subheader('Hyperparameter tuning to get better scores')
    st.write('Para fazer o ajuste dos hiperpar??metros existem algumas fun????es, a mais conhecida delas ?? o GridSearch do'
             ' Scikit-Learn, que vai procurar por todas as combina????es poss??veis de hiperpar??metros que o usu??rio '
             'passar, por??m acaba sendo muito custoso computacionalmente, sendo mais vantajoso utilizar o RandomSearch,'
             'tamb??m do Scikit, esse m??todo pode n??o obter o melhor conjunto poss??vel de hiperpar??metros, mas retornar??'
             ' um muito bom, em um tempo menor. Al??m desses dois citados, existem algumas bibliotecas especializadas em'
             'tuning de hiperpar??metros, entre elas a biblioteca Optuna, essa biblioteca consegue de forma bem mais '
             'eficiente encontrar os melhores hiperpar??metros.')
    toc.subheader2('Using optuna to find the best hyperparameters for Random Forest')
    st.write('Para usar o Optuna, ?? necess??rio criar uma fun????o objetivo, apontando todos os hiperpar??metros de '
             'interesse e o m??todo de avalia????o do modelo. Para o modelo Random Forest ser??o ajustados os '
             'hiperpar??metros \'n_estimators\', \'max_depth\', \'max_features\', \'min_samples_split\' e '
             '\'min_samples_leaf\'. Para avaliar o modelo ser?? usado o m??todo de Cross Validation usando o F1 score '
             'como m??trica de desempenho. Por fim, est?? sendo avaliado se as ??rvores de decis??o da Random Forest est??o '
             'criando "galhos" desnecess??rios, que podem trazer um sobreajuste para o modelo, para lidar com esses '
             'galhos, est?? sendo usado o m??todo de Prune (Podar), que descarta essas Florestas com overfitting.')
    ch_objrf = st.checkbox(label='Show objective_rf code')
    if ch_objrf:
        st.code(objective_rf_code)
    st.write('Agora basta criar um novo estudo do optuna.')
    st.code('study_rf = optuna.create_study(direction="maximize")\n'
            'study_rf.optimize(objective_rf, n_trials=500, timeout=300)')
    st.write(f'N??mero de trials conclu??das: {len(study_rf.trials)}')
    st.write(pruned)
    st.write(complete)
    st.write('Best Trial: ')
    st.write(val_best)
    st.write('  Params: ')
    for key, value in trial_rf.params.items():
        st.write("    {}: {}".format(key, value))
    st.plotly_chart(optimization_hist_rf)
    st.plotly_chart(parallel_coordinate_rf)

    st.write('Agora possuindo os melhores hiperpar??metros, pode-se construir o melhor modelo de Random Forest (Nesse '
             'modelo, fiz uns testes mantendo alguns dos hiperpar??metros e alterando o \'n_estimators\' e o '
             '\'min_samples_leaf\', consegui obter o melhor resultado.)')
    st.code('clf_rf_best = RandomForestClassifier(n_estimators=140, max_depth=12, max_features=\'auto\', '
            'min_samples_split=3, min_samples_leaf=2, random_state=5)\n'
            'clf_rf_best.fit(X_train_res_t,y_train_res_t)')
    st.code('y_pred_b = clf_rf_best.predict(X_test_t)\n'
            'classification_report(y_test_t, y_pred_b)')
    st.table(pd.DataFrame(report_rf_best).T)
    st.write('Com o ajuste dos hiperpar??metros o F1 Score foi melhorado, em consequ??ncia do aumento da precis??o em '
             'rela????o a Random Forest ajustada usando o atributo \'time\' mas sem ajuste de hiperpar??metros.')
    toc.subheader3('Confusion Matrix for Best Random Forest Model')
    st.pyplot(figrfbestcm, transparent=True)
    toc.subheader3('ROC curve for Best Random Forest Model')
    st.pyplot(figrocrfbest, transparent=True)
    st.write('Com o ajuste de hiperpar??metros e boa sele????o de features foi poss??vel aumentar o AUC score em 7% em '
             'rela????o ao modelo Random Forest inicial.')
    toc.subheader3('Feature importances for Best Random Forest Model')
    st.write('Ordem de import??ncia dos atributos usando a melhor Random Forest:')
    for key,value in zip(sort_features_rf_best.keys(), sort_features_rf_best.values()):
        st.write(f'{key}:',value)

    toc.subheader('Evaluating other models')
    st.write('Agora ser??o usados 2 diferentes modelos, para observar se h?? alguma melhora em rela????o aos modelos '
             'anteriores. O primeiro ser?? um classificador usando Support Vector Machine, o segundo ser?? um modelo '
             'usando o Extreme Gradient Boosting (XGBoost)')
    toc.subheader2('Using SVC to predict the target')
    st.write('Para o SVC ?? necess??rio usar o conjunto de dados normalizado, pois as SVM s??o sens??veis a atributos em '
             'diferentes escalas.')
    st.markdown('* Sem o atributo \'time\'')
    st.code('from sklearn.svm import SVC\n'
            'svm = SVC(random_state=5)\n'
            'svm.fit(X_train_s_res, y_train_s)')
    st.code('y_pred = svm.predict(X_test_s)\n'
            'classification_report(y_test, y_pred)')
    st.table(pd.DataFrame(report_svm).T)
    st.markdown('* Com o atributo \'time\'')
    st.code('svm.fit(X_train_st_res, y_train_st)\n'''
            'y_pred = svm.predict(X_test_st)\n'
            'classification_report(y_test_t, y_pred)')
    st.table(pd.DataFrame(report_svm_time).T)
    st.write('O modelo SVC sem a feature \'time\' performou melhor que a Decision Tree e praticamente igual a '
             'Random Forest inicial. J?? com a feature \'time\', obteve o mesmo resultado que a Random Forest com esse '
             'atributo. Sendo assim, modelos usando SVM s??o potenciais classificadores para esse problema.')

    toc.subheader2('Using XGBoost')
    st.write('O modelo XGBoost ?? um modelo ensemble, ou seja, junta um conjunto de modelos para gerar o resultado, '
             'ele utiliza o m??todo de Boosting de gradiente para conseguir atingir melhores resultados. Atualmente ?? c'
             'onsiderado o \'state-of-the-art\' para conjuntos de dados tabulares')
    st.code('from xgboost import XGBClassifier\n'
            'xgb = XGBClassifier(learning_rate=0.01, n_estimators=180, random_state=5)\n'
            'xgb.fit(X_train_res_t, y_train_res_t)')
    st.code('y_pred = xgb.predict(X_test_t)\n'
            'classification_report(y_test_t, y_pred)')
    st.table(pd.DataFrame(report_xgb).T)
    st.write('Sem tuning de hiperpar??metros o modelo XGBoost obteve resultados similares ao SVM.')

    toc.subheader3('Tuning XGBoost hyperparameters')
    st.write('Assim como para a Random Forest, ?? necess??rio criar uma fun????o objetivo para o XGBoost. Nessa fun????o '
             'objetivo foi utilizada a m??trica F1 Score.')
    ch_objxgb = st.checkbox(label='Show Objective function for XGBoost')
    if ch_objxgb:
        st.code(objective_code)
    st.write('Agora basta criar um novo estudo do optuna para o XGBoost.')
    st.code('study = optuna.create_study(direction="maximize")\n'
            'study.optimize(objective_rf, n_trials=500, timeout=300)')
    st.write(finished)
    st.write('Best Trial: ')
    st.write(Value)
    st.write(' Params:')
    for key, value in trial.params.items():
        st.write("    {}: {}".format(key, value))
    st.plotly_chart(optimization_hist_xgb)
    st.plotly_chart(parallel_coordinate_xgb)
    st.write('Agora pode-se construir o melhor modelo usando XGBoost')
    st.code('best_param_xgb = trial.params\n'
            'best_param_xgb["verbosity"] = 0\n'
            'best_param_xgb["objective"] = "binary:logistic"\n'
            'best_param_xgb["tree_method"] = "exact"')
    st.code('xgb_best = XGBClassifier(**best_param_xgb)\n'
            'xgb_best.fit(X_train_res_t,y_train_res_t)')
    st.code('y_pred_best = xgb_best.predict(X_test_t)\n'
            'classification_report(y_test_t,y_pred_best)')
    st.table(pd.DataFrame(report_xgb_best).T)
    st.write('Houve melhora significativa, tanto para a precis??o como para o recall, gerando um F1 Score tamb??m melhor,'
             ' e, consequentemente, uma acur??cia melhor.')
    toc.subheader3('Confusion Matrix for Best XGBoost Model')
    st.pyplot(figxgbbestcm, transparent=True)
    toc.subheader3('ROC curve for Best XGBoost Model')
    st.pyplot(figrocxgbbest, transparent=True)
    st.write('Como esperado, o XGBoost foi o modelo que se saiu melhor entre todos os outros, conseguindo ficando na '
             'frente do melhor modelo de Random Forest em 3%. Com mais tempo, ?? poss??vel aprimor??-lo ainda mais, com '
             't??cnicas de feature engineering e processamento dos dados.')
    toc.subheader3('Feature importances for Best XGBoost Model')
    st.write('Ordem de import??ncia dos atributos usando o melhor XGBoost:')
    for key,value in zip(sort_features_xgb_best.keys(), sort_features_xgb_best.values()):
        st.write(f'{key}:',value)

if check_box_conc:
    toc.header('Conclusion')
    st.markdown('---')
    st.write('Como foi mostrado inicialmente, as features com maiores correla????es com a vari??vel de sa??da estiveram '
             'sempre no topo de import??ncia para os modelos testados. Sendo elas: \'time\', \'serum_creatinine\', '
             '\'ejection_fraction\', \'age\' e \'serum_sodium\'. Removendo o atributo \'time\' dessa lista, ficamos '
             'apenas com atributos bi??logicos, sendo assim, pode ser realizado um estudo mais aprofundado acerca desses'
             'atributos para analisar o quanto eles realmente impactam na insufici??ncia card??aca, podendo-se criar '
             'meios de tratamento que lidem diretamente com esses atributos.')
    st.write('')
    st.write('Obrigado!')
    st.write('Trabalho Desenvolvido por Gabriel Barros Lins')




toc.generate()


#
# st.markdown("---")
# st.markdown("## Descri????o")
# if check_box_col:
#     st.markdown(markdown_1)
#
# st.markdown("---")
# st.markdown("## Visualiza????o dos Dados")
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

