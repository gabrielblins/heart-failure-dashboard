import streamlit as st

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('DashBoard de análise de empréstimos')
st.write('Esse prejeto consiste na analise de dados relativo à empréstimos no qual tentamos visualizar relações entre features que possuimos')

st.sidebar.title("Configurações")
check_box_eda = st.sidebar.checkbox(label="Mostrar Exploratory Data Analysis")
check_box_col = st.sidebar.checkbox(label="Mostrar Descrição de colunas")
check_box_noise = st.sidebar.checkbox(label="Mostrar Removing noise from data")
check_box_tv = st.sidebar.checkbox(label="Mostrar Training and Validation")
check_box_bn = st.sidebar.checkbox(label="Mostrar Bonus Section")

if check_box_col:

if check_box_eda:

if check_box_noise:

if check_box_tv:

if check_box_bn:

st.markdown("---")
st.markdown("## Descrição")
if check_box_col:
    st.markdown(markdown_1)

st.markdown("---")
st.markdown("## Visualização dos Dados")

st.markdown(markdown_2)
st.plotly_chart(fig1, use_container_width=True)
st.markdown(markdown_3)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(markdown_4)
st.plotly_chart(fig3, use_container_width=True)
st.markdown(markdown_5)
st.plotly_chart(fig4, use_container_width=True)
st.plotly_chart(fig5, use_container_width=True)
st.markdown(markdown_7)
st.plotly_chart(fig6, use_container_width=True)
st.markdown(markdown_6)
st.plotly_chart(fig7, use_container_width=True)
st.plotly_chart(fig8, use_container_width=True)

st.markdown(markdown_8, unsafe_allow_html=True)
