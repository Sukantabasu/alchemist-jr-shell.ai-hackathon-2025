import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.title("ALCHEMIST - Sample Input")

# Introduction text
st.markdown("""
Welcome to ALCHEMIST, an advanced machine learning framework for predicting fuel blend properties. 
This sophisticated system employs a multi-stage ensemble approach combining AutoGluon, RealMLP, and TabPFN models.
Our methodology processes component fractions and properties through engineered features to deliver accurate predictions.
The framework utilizes out-of-fold predictions as additional features to capture complex inter-target correlations.
Enter your component data below to experience the power of our predictive alchemical transformation.
""")

# Get the directory where the script is located
APP_DIR = Path(__file__).parent
IMG_PATH = APP_DIR / "Framework.png"
TEST_CSV_PATH = APP_DIR / "test.csv"
TRAIN_CSV_PATH = APP_DIR / "train_processed.csv"

# Display the framework images
col1, col2 = st.columns([1, 2])

with col1:
    st.image(str(IMG_PATH), caption="ALCHEMIST Framework Architecture",
             use_container_width=True)

with col2:
    st.image(str(IMG_PATH), caption="ALCHEMIST Implementation Details",
             use_container_width=True)

# Initialize all session state variables
if 'property_data' not in st.session_state:
    st.session_state.property_data = {}
    for comp in range(1, 6):
        for prop in range(1, 11):
            st.session_state.property_data[
                f'Component{comp}_Property{prop}'] = 0.0

if 'component_fractions' not in st.session_state:
    st.session_state.component_fractions = [0.0, 0.0, 0.0, 0.0, 0.0]

if 'feature_results' not in st.session_state:
    st.session_state.feature_results = None

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# File upload section
st.subheader("Load Data File")

# Add option to load default test set
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Upload CSV or Excel file",
                                     type=['csv', 'xlsx'])

with col2:
    if st.button("Load Demo Data"):
        # Load default test data from test.csv (invisible to user)
        try:
            default_df = pd.read_csv(str(TEST_CSV_PATH))
            if len(default_df) > 0:
                st.session_state.uploaded_df = default_df
                st.session_state.available_ids = default_df['ID'].tolist()

                # Auto-load the first row
                selected_row = default_df.iloc[0]

                # Load component fractions
                for comp in range(1, 6):
                    fraction_col = f'Component{comp}_fraction'
                    if fraction_col in selected_row:
                        st.session_state.component_fractions[comp - 1] = \
                        selected_row[fraction_col]

                # Load property values
                for comp in range(1, 6):
                    for prop in range(1, 11):
                        prop_col = f'Component{comp}_Property{prop}'
                        if prop_col in selected_row:
                            st.session_state.property_data[
                                f'Component{comp}_Property{prop}'] = \
                            selected_row[prop_col]

                st.success(
                    f"Demo data loaded! (Sample ID: {selected_row['ID']})")
                st.rerun()
            else:
                st.error("Demo data file is empty.")
        except FileNotFoundError:
            st.error("Demo data not available.")
        except Exception as e:
            st.error(f"Error loading demo data: {str(e)}")

if uploaded_file is not None:
    if st.button("Load and Process File"):
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write(f"File loaded! Found {len(df)} rows.")
        st.session_state.uploaded_df = df

        if 'ID' in df.columns:
            st.session_state.available_ids = df['ID'].tolist()
            st.success("File processed successfully! Now select an ID below.")
        else:
            st.error("No 'ID' column found in the uploaded file.")

# Show ID selection only after file is processed
if 'uploaded_df' in st.session_state and 'available_ids' in st.session_state:
    selected_id = st.selectbox("Select ID:", st.session_state.available_ids)

    if st.button("Load Selected Row Data"):
        df = st.session_state.uploaded_df
        selected_row = df[df['ID'] == selected_id].iloc[0]

        # Load component fractions
        for comp in range(1, 6):
            fraction_col = f'Component{comp}_fraction'
            if fraction_col in selected_row:
                st.session_state.component_fractions[comp - 1] = selected_row[
                    fraction_col]

        # Load property values
        for comp in range(1, 6):
            for prop in range(1, 11):
                prop_col = f'Component{comp}_Property{prop}'
                if prop_col in selected_row:
                    st.session_state.property_data[
                        f'Component{comp}_Property{prop}'] = selected_row[
                        prop_col]

        st.success(f"Data loaded for ID {selected_id}!")
        st.rerun()

# Component fractions section
st.subheader("Component Fractions")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.session_state.component_fractions[0] = st.number_input("Component1",
                                                              value=
                                                              st.session_state.component_fractions[
                                                                  0],
                                                              format="%.3f",
                                                              key="frac1")
with col2:
    st.session_state.component_fractions[1] = st.number_input("Component2",
                                                              value=
                                                              st.session_state.component_fractions[
                                                                  1],
                                                              format="%.3f",
                                                              key="frac2")
with col3:
    st.session_state.component_fractions[2] = st.number_input("Component3",
                                                              value=
                                                              st.session_state.component_fractions[
                                                                  2],
                                                              format="%.3f",
                                                              key="frac3")
with col4:
    st.session_state.component_fractions[3] = st.number_input("Component4",
                                                              value=
                                                              st.session_state.component_fractions[
                                                                  3],
                                                              format="%.3f",
                                                              key="frac4")
with col5:
    st.session_state.component_fractions[4] = st.number_input("Component5",
                                                              value=
                                                              st.session_state.component_fractions[
                                                                  4],
                                                              format="%.3f",
                                                              key="frac5")
with col6:
    total_fraction = sum(st.session_state.component_fractions)
    st.metric("Total", f"{total_fraction:.3f}")
    if abs(total_fraction - 1.0) > 0.001:
        st.error("⚠️ Total must equal 1.0")

# Property input section
st.subheader("Component Properties")

if 'uploaded_df' in st.session_state:
    st.write("**Use sliders to modify property values:**")

    # Color styles for different properties
    color_styles = [
        "background-color: #FF6B6B; color: white; padding: 2px 8px; border-radius: 4px;",
        "background-color: #4ECDC4; color: white; padding: 2px 8px; border-radius: 4px;",
        "background-color: #45B7D1; color: white; padding: 2px 8px; border-radius: 4px;",
        "background-color: #96CEB4; color: white; padding: 2px 8px; border-radius: 4px;",
        "background-color: #FFEAA7; color: black; padding: 2px 8px; border-radius: 4px;",
        "background-color: #DDA0DD; color: white; padding: 2px 8px; border-radius: 4px;",
        "background-color: #98D8C8; color: white; padding: 2px 8px; border-radius: 4px;",
        "background-color: #F7DC6F; color: black; padding: 2px 8px; border-radius: 4px;",
        "background-color: #BB8FCE; color: white; padding: 2px 8px; border-radius: 4px;",
        "background-color: #85C1E9; color: white; padding: 2px 8px; border-radius: 4px;"
    ]

    # Show all properties with colored headers
    for prop in range(1, 11):
        st.markdown(
            f'<span style="{color_styles[prop - 1]}">Property {prop}</span>',
            unsafe_allow_html=True)
        cols = st.columns(5)
        for comp in range(1, 6):
            with cols[comp - 1]:
                property_key = f"Component{comp}_Property{prop}"
                current_val = st.session_state.property_data[property_key]

                new_val = st.slider(
                    f"C{comp}P{prop}",
                    min_value=-3.0,
                    max_value=3.0,
                    value=float(current_val),
                    step=0.1,
                    key=f"slider_{property_key}",
                    help=f"Component {comp} Property {prop}"
                )
                st.session_state.property_data[property_key] = new_val
else:
    st.info("Please upload and load a file to modify property values.")

# Property values visualization
st.subheader("All Properties Across Components")

chart_data = []
for prop in range(1, 11):
    for comp in range(1, 6):
        comp_key = f"Component{comp}_Property{prop}"
        chart_data.append({
            'Property': f"P{prop}",
            'Component': f"C{comp}",
            'Value': st.session_state.property_data[comp_key]
        })

df_chart = pd.DataFrame(chart_data)
fig = px.bar(df_chart, x='Property', y='Value', color='Component',
             barmode='group')
fig.update_layout(title="Property Values by Component", font=dict(size=12))
st.plotly_chart(fig, use_container_width=True)

# Feature Engineering section
st.subheader("Feature Engineering")
if st.button("Generate Engineered Features"):

    # Create volume fraction-weighted features
    contributions_data = []
    for prop in range(1, 11):
        for comp in range(1, 6):
            fraction = st.session_state.component_fractions[comp - 1]
            property_key = f"Component{comp}_Property{prop}"
            property_value = st.session_state.property_data[property_key]
            contribution = fraction * property_value
            contributions_data.append({
                'Feature': f'C{comp}_Contribution_P{prop}',
                'Value': contribution
            })

    contributions_df = pd.DataFrame(contributions_data)

    # Create weighted-averaged features
    weighted_avg_data = []
    for prop in range(1, 11):
        weighted_sum = 0
        for comp in range(1, 6):
            fraction = st.session_state.component_fractions[comp - 1]
            property_key = f"Component{comp}_Property{prop}"
            property_value = st.session_state.property_data[property_key]
            weighted_sum += fraction * property_value

        weighted_avg_data.append({
            'Feature': f'WeightedAvg_P{prop}',
            'Value': weighted_sum
        })

    weighted_avg_df = pd.DataFrame(weighted_avg_data)

    # Store results in session state
    st.session_state.feature_results = {
        'contributions': contributions_df,
        'weighted_avg': weighted_avg_df
    }

# Display feature engineering results
if st.session_state.feature_results is not None:
    col1, col2, col3 = st.columns([3, 3, 4])

    with col1:
        st.write("**Volume Fraction-Weighted Features:**")
        st.dataframe(
            st.session_state.feature_results['contributions'],
            use_container_width=True,
            hide_index=True,
            height=400
        )

    with col2:
        st.write("**Weighted-Averaged Features:**")
        st.dataframe(
            st.session_state.feature_results['weighted_avg'],
            use_container_width=True,
            hide_index=True,
            height=400
        )

    with col3:
        # Add some vertical padding to center the chart
        st.markdown('<div style="margin-top: 40px;">', unsafe_allow_html=True)

        # Create horizontal bar chart
        fig_eng = px.bar(
            st.session_state.feature_results['weighted_avg'],
            x='Value',
            y='Feature',
            orientation='h'
        )

        # Reverse the y-axis order to show P1 at top, P10 at bottom
        fig_eng.update_layout(yaxis={'categoryorder': 'array',
                                     'categoryarray': ['WeightedAvg_P10',
                                                       'WeightedAvg_P9',
                                                       'WeightedAvg_P8',
                                                       'WeightedAvg_P7',
                                                       'WeightedAvg_P6',
                                                       'WeightedAvg_P5',
                                                       'WeightedAvg_P4',
                                                       'WeightedAvg_P3',
                                                       'WeightedAvg_P2',
                                                       'WeightedAvg_P1']})

        fig_eng.update_traces(
            marker=dict(color='#3498db', opacity=0.8)
        )

        fig_eng.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            xaxis_title="Value",
            yaxis_title="",
            font=dict(size=10),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinecolor='rgba(128,128,128,0.5)'
            ),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white'
        )

        st.plotly_chart(fig_eng, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
st.subheader("ALCHEMIST Lightweight Predictions")

# Target selection
target_names = [f"BlendProperty{i + 1}" for i in range(10)]
selected_target = st.selectbox("Select Target to Predict:", target_names)
selected_target_idx = target_names.index(selected_target)

# Training sample size selection
num_samples = st.slider("Number of Training Samples", min_value=10,
                        max_value=1000, value=100, step=10)
if num_samples > 500:
    st.warning(
        "⚠️ Using more than 500 samples may be slow. For fast computation, use fewer samples.")
if num_samples == 1000:
    st.error("⚠️ 1000 samples may cause timeout or performance issues!")

if st.button("Make Predictions with TabPFN"):

    # Prepare current sample data with engineered features
    current_sample = []

    # Add component fractions
    current_sample.extend(st.session_state.component_fractions)

    # Add contribution features
    for prop in range(1, 11):
        for comp in range(1, 6):
            fraction = st.session_state.component_fractions[comp - 1]
            property_key = f"Component{comp}_Property{prop}"
            property_value = st.session_state.property_data[property_key]
            contribution = fraction * property_value
            current_sample.append(contribution)

    # Add weighted average features
    for prop in range(1, 11):
        weighted_sum = 0
        for comp in range(1, 6):
            fraction = st.session_state.component_fractions[comp - 1]
            property_key = f"Component{comp}_Property{prop}"
            property_value = st.session_state.property_data[property_key]
            weighted_sum += fraction * property_value
        current_sample.append(weighted_sum)

    st.write(f"Total engineered features: {len(current_sample)}")

    try:
        # Load and shuffle training data, then select samples
        train_data_full = pd.read_csv(str(TRAIN_CSV_PATH))
        train_data_shuffled = train_data_full.sample(frac=1,
                                                     random_state=42).reset_index(
            drop=True)
        train_data = train_data_shuffled.head(num_samples)

        st.write(
            f"Training data loaded: {train_data.shape[0]} samples, 65 input features + 10 targets")

        # Prepare training features and selected target
        X_train = train_data.iloc[:, :65].values
        y_train = train_data.iloc[:, 65 + selected_target_idx].values

        # Make prediction using TabPFN
        from tabpfn import TabPFNRegressor

        current_sample_array = np.array(current_sample).reshape(1, -1)

        regressor = TabPFNRegressor()
        regressor.fit(X_train, y_train)
        prediction = regressor.predict(current_sample_array)[0]

        # Store prediction in history
        prediction_record = {
            'target': selected_target,
            'num_samples': num_samples,
            'prediction': prediction
        }
        st.session_state.prediction_history.append(prediction_record)

        # Display prediction
        st.success(f"**Prediction for {selected_target}: {prediction:.4f}**")

        # Show note about prediction history
        current_target_history = [record for record in
                                  st.session_state.prediction_history if
                                  record['target'] == selected_target]

        if len(current_target_history) == 1:
            st.info(
                "💡 Try making predictions with different numbers of training samples to see how predictions change!")

        # Plot prediction history for current target
        if len(current_target_history) > 1:
            st.subheader(f"Prediction History for {selected_target}")

            history_df = pd.DataFrame(current_target_history)

            fig_history = px.scatter(
                history_df,
                x='num_samples',
                y='prediction'
            )

            # Customize markers
            fig_history.update_traces(
                marker=dict(size=8, color='blue', opacity=0.8)
            )

            fig_history.update_layout(
                xaxis_title="Number of Training Samples",
                yaxis_title="Prediction Value",
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=20, b=0)
            )

            st.plotly_chart(fig_history, use_container_width=True)

    except FileNotFoundError:
        st.error(
            "train_processed.csv file not found. Please ensure the file is in the same directory.")
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")