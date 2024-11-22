import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import optimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import glob
import re
import os


class XY_Magtorq():
    def __init__(self):
        # Read all data files
        self.temperature_data = pd.read_csv('./data/temperature.csv')
        self.initial_resistance_data = pd.read_csv('./data/resistance.csv')
        self.voltage_data = pd.read_csv('./data/voltage_max.csv')

        # Extract measured data
        measured_temps = self.temperature_data['T (C)'].values
        R_ref = self.initial_resistance_data['R (Ω)'].iloc[0]
        V_max = self.voltage_data['V_bat (V)'].iloc[0]

        self.R_ref = R_ref                          # @ reference temperature
        self.V_max = V_max
        self.I_max = V_max / R_ref
        self.P_max = (V_max ** 2) / R_ref
        self.T_ref = measured_temps[0]
        self.RHO = 0.00393                      # copper resistivity (ohm/m)
        self.magnetic_moment = None

        properties = {
            # Electrical Properties
            'resistance': {
                'value': format(float(self.R_ref), '.2e'),
                'units': 'ohms'
            },
            'voltage_max': {
                'value': format(float(self.V_max), '.2e'),
                'units': 'V'
            },
            'current_max': {
                'value': format(float(self.I_max), '.2e'),
                'units': 'A'
            },
            'power_max': {
                'value': format(float(self.P_max), '.2e'),
                'units': 'W'
            },
            
            # Material Properties
            'resistivity': {  
                'value': format(float(self.RHO), '.2e'),
                'units': 'ohms*m'
            },
            
            # Reference Conditions
            'temperature_ref': {
                'value': format(float(self.T_ref), '.2e'),  
                'units': 'C'  
            }
        }
    
        with open('properties.json', 'w') as f:
            json.dump(properties, f, indent=4)

    def find_moment(self):
        # Physical constants
        μ0 = 4 * np.pi * 1e-7  # permeability of free space in H/m

        def load_and_process_csv(filepath: str) -> pd.DataFrame:
            """Load a CSV file and process it into a clean DataFrame."""
            df = pd.read_csv(filepath)
            # Calculate total magnetic field if not already present
            if 'M(µT)' not in df.columns:
                df['M(µT)'] = np.sqrt(df['M_x(µT)']**2 + df['M_y(µT)']**2 + df['M_z(µT)']**2)
            return df

        def extract_distance_from_filename(filename: str) -> int:
            """Extract the distance in cm from the filename (e.g., I5-1.csv -> 5)."""
            match = re.search(r'[IO](\d+)-', filename)
            if match:
                return int(match.group(1))
            raise ValueError(f"Could not extract distance from filename: {filename}")

        def calculate_average_field(df: pd.DataFrame) -> float:
            """Calculate the average total magnetic field from a DataFrame."""
            return df['M(µT)'].mean()

        def subtract_ambient_field(measurement_df: pd.DataFrame, ambient_df: pd.DataFrame) -> float:
            """
            Calculate the difference between measurement and ambient magnetic fields
            using vector subtraction to account for possible different orientations.
            
            Returns the RMS of the difference vector magnitude over time.
            """
            # Calculate mean ambient field components
            ambient_mean = {
                'x': ambient_df['M_x(µT)'].mean(),
                'y': ambient_df['M_y(µT)'].mean(),
                'z': ambient_df['M_z(µT)'].mean()
            }
            
            # Subtract ambient field components from measurement
            diff_x = measurement_df['M_x(µT)'] - ambient_mean['x']
            diff_y = measurement_df['M_y(µT)'] - ambient_mean['y']
            diff_z = measurement_df['M_z(µT)'] - ambient_mean['z']
            
            # Calculate magnitude of difference vector
            diff_magnitude = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
            
            # Return RMS of the difference magnitude
            return float(np.sqrt(np.mean(diff_magnitude**2)))

        def calculate_magnetic_moment(field: float, distance: float) -> float:
            """
            Calculate the magnetic moment using the dipole formula:
            B = (μ0/4π) * (3(m·r)r/r^5 - m/r^3)
            
            For our simplified case along the axis of the dipole:
            B = (μ0/2π) * (m/r^3)
            
            Therefore:
            m = (2π/μ0) * B * r^3
            """
            distance_m = distance / 100  # convert cm to meters
            field_T = field * 1e-6  # convert μT to T
            
            # Calculate magnetic moment in A·m²
            moment = (2 * np.pi / μ0) * field_T * (distance_m ** 3)
            
            return moment
        
        # Load ambient field data
        ambient_df = load_and_process_csv('./data/magnetic-field/ambient-field.csv')
        ambient_field = calculate_average_field(ambient_df)
        
        # Process all measurement files
        distances = []
        net_fields = []
        magnetic_moments = []
        
        # Find all I*-*.csv files
        measurement_files = glob.glob('./data/magnetic-field/I*-*.csv')
        measurement_files.extend(glob.glob('./data/magnetic-field/O*-*.csv'))
        
        for file in measurement_files:
            try:
                # Extract distance and load data
                distance = extract_distance_from_filename(file)
                df = load_and_process_csv(file)
                
                # Calculate net field by vector subtraction from ambient
                net_field = subtract_ambient_field(df, ambient_df)
                
                # Calculate magnetic moment
                moment = calculate_magnetic_moment(net_field, distance)
                
                # Store results
                distances.append(distance)
                net_fields.append(net_field)
                magnetic_moments.append(moment)
                
                print(f"Processed {file}: distance={distance}cm, net_field={net_field:.2f}μT, magnetic_moment={moment:.2f}Am^2")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        
        # Compute mean moment
        avg_moment = np.mean(magnetic_moments)
        
        moment_json = {
            'magnetic_moment': {
                'value': format(float(avg_moment), '.2e'),
                'units': 'Am^2',
            }
        }

        self.magnetic_moment = avg_moment
        
        with open('properties.json', 'r+') as f:
            properties = json.load(f)
            properties.update(moment_json)
            f.seek(0)
            json.dump(properties, f, indent=4)
            f.truncate()
        
        print(f"\nAnalysis complete!")
        print(f"Average magnetic moment: {avg_moment:.2e} A·m²")
        print(f"Results saved to properties.json")

    def find_thermals(self):
        # Extract measured data
        measured_temps = self.temperature_data['T (C)'].values
        measured_times = self.temperature_data['t'].values

        # Thermal model function (exponential heating)
        def thermal_model(t, T_init, T_final, tau):
            return T_init + (T_final - T_init) * (1 - np.exp(-t/tau))

        # Fit thermal model to measured data
        def fit_thermal_model(t, T_final, tau):
            return thermal_model(t, measured_temps[0], T_final, tau)

        # Fit the thermal model
        params, params_covariance = optimize.curve_fit(
            fit_thermal_model,
            measured_times,
            measured_temps,
            p0=[45, 300]  # Initial guesses for T_final and tau
        )

        T_final_fit, tau_fit = params
        print(f"Fitted thermal parameters:")
        print(f"Final temperature: {T_final_fit:.2f}°C")
        print(f"Time constant: {tau_fit:.1f} seconds")

        # Create extended time array (1 hour = 3600 seconds)
        extended_times = np.linspace(0, 3600, 1000)
        predicted_temps = thermal_model(extended_times, measured_temps[0], T_final_fit, tau_fit)

        # Calculate resistance model
        def theoretical_resistance(T, R_ref, alpha, T_ref=20):
            return R_ref * (1 + alpha * (T - T_ref))

        # Fit the resistance model to get actual alpha
        def fit_function(T, alpha):
            return theoretical_resistance(T, self.R_ref, alpha)

        params, params_covariance = optimize.curve_fit(
            fit_function, 
            measured_temps, 
            theoretical_resistance(measured_temps, self.R_ref, self.RHO)
        )

        fitted_alpha = params[0]
        print(f"Fitted temperature coefficient (alpha): {fitted_alpha:.6f} /°C")

        # Create extended temperature range for parameter analysis (-50°C to 150°C)
        extended_temp_range = np.linspace(-50, 150, 1000)
        theoretical_resistances = theoretical_resistance(extended_temp_range, self.R_ref, fitted_alpha)
        theoretical_currents = self.V_max / theoretical_resistances
        theoretical_moments = self.magnetic_moment * (theoretical_currents / (self.V_max / self.R_ref))

        # Create figure
        fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Temperature vs Time',
                                        'Resistance vs Temperature (-50°C to 150°C)',
                                        'Current vs Temperature (-50°C to 150°C)',
                                        'Magnetic Moment vs Temperature (-50°C to 150°C)'),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.12)

        # Temperature vs Time
        fig.add_trace(
            go.Scatter(x=measured_times, y=measured_temps,
                    mode='markers',
                    name='Measured Temperature',
                    marker=dict(color='orange', size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=extended_times, y=predicted_temps,
                    mode='lines',
                    name='Predicted Temperature',
                    line=dict(color='orange', width=2)),
            row=1, col=1
        )

        # Resistance Model
        fig.add_trace(
            go.Scatter(x=measured_temps, y=theoretical_resistance(measured_temps, self.R_ref, fitted_alpha),
                    mode='markers',
                    name='Measured R',
                    marker=dict(color='red', size=8)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=extended_temp_range, y=theoretical_resistances,
                    mode='lines',
                    name='R(T) Model',
                    line=dict(color='red', width=2)),
            row=1, col=2
        )

        # Current Model
        fig.add_trace(
            go.Scatter(x=measured_temps, y=self.V_max/theoretical_resistance(measured_temps, self.R_ref, fitted_alpha),
                    mode='markers',
                    name='Measured I',
                    marker=dict(color='blue', size=8)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=extended_temp_range, y=theoretical_currents,
                    mode='lines',
                    name='I(T) Model',
                    line=dict(color='blue', width=2)),
            row=2, col=1
        )

        # Magnetic Moment Model
        fig.add_trace(
            go.Scatter(x=measured_temps, y=self.magnetic_moment * (self.V_max/theoretical_resistance(measured_temps, self.R_ref, fitted_alpha))/(self.V_max/self.R_ref),
                    mode='markers',
                    name='Measured M',
                    marker=dict(color='green', size=8)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=extended_temp_range, y=theoretical_moments,
                    mode='lines',
                    name='M(T) Model',
                    line=dict(color='green', width=2)),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            title_text=f'Extended Thermal Analysis (α = {fitted_alpha:.6f} /°C)',
            template='plotly_white',
            showlegend=True
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
        fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_xaxes(title_text="Temperature (°C)", row=2, col=2)

        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Resistance (Ω)", row=1, col=2)
        fig.update_yaxes(title_text="Current (A)", row=2, col=1)
        fig.update_yaxes(title_text="Magnetic Moment (Am²)", row=2, col=2)

        # Add grids to all subplots
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=i, col=j)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=i, col=j)

        # Set temperature range for parameter plots
        for i in range(1, 3):
            for j in range(1, 3):
                if not (i == 1 and j == 1):  # Skip the time series plot
                    fig.update_xaxes(range=[-50, 100], row=i, col=j)

        # Save the plot
        fig.write_html("./plots/extended_thermal_analysis.html", 
                        auto_open=True,
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
                        }
        )

        # Print analysis results
        print("\nProjected Values at 1 hour:")
        T_1hour = thermal_model(3600, measured_temps[0], T_final_fit, tau_fit)
        R_1hour = theoretical_resistance(T_1hour, self.R_ref, fitted_alpha)
        I_1hour = self.V_max / R_1hour
        M_1hour = self.magnetic_moment * (I_1hour / (self.V_max / self.R_ref))

        print(f"Temperature: {T_1hour:.1f}°C")
        print(f"Resistance: {R_1hour:.3f}Ω")
        print(f"Current: {I_1hour:.3f}A")
        print(f"Magnetic Moment: {M_1hour:.3e}Am²")

        # Print values at extreme temperatures
        print("\nValues at Temperature Extremes:")
        for T in [-50, 150]:
            R = theoretical_resistance(T, self.R_ref, fitted_alpha)
            I = self.V_max / R
            M = self.magnetic_moment * (I / (self.V_max / self.R_ref))
            print(f"\nAt {T}°C:")
            print(f"Resistance: {R:.3f}Ω")
            print(f"Current: {I:.3f}A")
            print(f"Magnetic Moment: {M:.3e}Am²")




board = XY_Magtorq()
board.find_moment()
board.find_thermals()