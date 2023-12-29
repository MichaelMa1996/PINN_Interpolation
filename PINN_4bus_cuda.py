import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import os
import random
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)


reset_random_seeds()


def find_indices(array):
    indices = []
    for j in range(array.shape[1]):
        for i in range(array.shape[0]):
            if abs(array[i, j]) > 0.001:
                indices.append((i, j))
    return indices

def Find_derivative(resolution, u_bc):
    Time_step = 1/ (resolution)
    # Calculate the first derivative manually for each feature
    first_derivative_manual = np.zeros_like(u_bc)
    for j in range(u_bc.shape[1]):
        for i in range(1, u_bc.shape[0]-1):
            first_derivative_manual[i, j] = (u_bc[i+1, j] - u_bc[i-1, j]) / (2 * Time_step)

    # Calculate the second derivative manually for each feature
    second_derivative_manual = np.zeros_like(u_bc)
    for j in range(u_bc.shape[1]):
        for i in range(1, u_bc.shape[0]-1):
            second_derivative_manual[i, j] = (u_bc[i+1, j] - 2 * u_bc[i, j] + u_bc[i-1, j]) / (Time_step ** 2)
    a=1
    return first_derivative_manual, second_derivative_manual

def evenly_select_data(data, time, num_points):
    if num_points >= len(data):
        return data, time

    indices = np.linspace(0, len(data) - 1, num_points, dtype=int)
    selected_data = data[indices]
    selected_time = time[indices]

    return selected_data, selected_time


def calculate_mape(actual_data, predicted_data, threshold=0.0005):
    mask = np.abs(actual_data) > threshold
    masked_actual = actual_data[mask]
    masked_predicted = predicted_data[mask]

    n = masked_actual.shape[0]
    absolute_percentage_errors = np.abs((masked_actual - masked_predicted) / masked_actual)
    mape = (1 / n) * np.sum(absolute_percentage_errors) * 100
    return mape


def randomize_values(value1, value2, value3, rate_of_change):
    modified_value1 = value1 + value1 * random.uniform(-rate_of_change, rate_of_change)
    modified_value2 = value2 + value2 * random.uniform(-rate_of_change, rate_of_change)
    modified_value3 = value3 + value3 * random.uniform(-rate_of_change, rate_of_change)

    return modified_value1, modified_value2, modified_value3


def calculate_mape2(actual, predicted):
    indices = find_indices(actual)
    actual1 = actual[indices]
    predicted1 = predicted[indices]
    absolute_error = np.abs(actual1 - predicted1.reshape(-1, 1))
    percentage_error = absolute_error / actual1
    mape = np.abs(np.mean(percentage_error) * 100)
    return mape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 80).to(device)
        self.hidden_layer2 = nn.Linear(80, 80).to(device)
        self.hidden_layer3 = nn.Linear(80, 80).to(device)
        self.hidden_layer4 = nn.Linear(80, 80).to(device)
        self.hidden_layer5 = nn.Linear(80, 80).to(device)
        self.output_layer = nn.Linear(80, 4).to(device)

    def forward(self, t):
        inputs = torch.cat([t], axis=1).to(device)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        return output


def Train_PINN_model(iterations, t_bc, u_bc, learning_rate, t):
    net = Net().to(device)
    mse_cost_function = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(iterations):
        optimizer.zero_grad()

        pt_t_bc = t_bc
        pt_u_bc = u_bc
        net_bc_out = net(pt_t_bc)
        mse_u = mse_cost_function(net_bc_out, pt_u_bc)

        pt_t_bc2 = time
        pt_u_bc2 = torch.from_numpy(delta_pool[:, :2]).float().to(device)
        net_bc_out2 = net(pt_t_bc2)[:, :2]
        mse_u2 = mse_cost_function(net_bc_out2, pt_u_bc2)

        t_collocation = t
        all_zeros = torch.zeros_like(t_collocation)
        pt_t_collocation = t_collocation.requires_grad_(True).to(device)
        pt_all_zeros = all_zeros
        f_out = ode_function(pt_t_collocation, net)
        mse_f = mse_cost_function(f_out, pt_all_zeros)

        loss = mse_f + mse_u

        loss.backward()
        optimizer.step()

        with torch.autograd.no_grad():
            print(epoch, "Total Training Loss:", loss.item(), "ODE Loss:", mse_f.item(), "Boundary Loss:", mse_u.item())

    return net


def ode_function(t, net):
    u = net(t)
    ode = 0
    dot_u = torch.empty((u.shape[0], 0), dtype=torch.float32).to(device)
    double_dot_u = torch.empty((u.shape[0], 0), dtype=torch.float32).to(device)
    for i in range(u.shape[1]):
        dot_u = torch.column_stack((dot_u, torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0]))
        double_dot_u = torch.column_stack(
            (double_dot_u, torch.autograd.grad(dot_u[:, i].sum(), t, create_graph=True)[0]))

    # Now convert your PyTorch tensors to numpy arrays to use with scikit-learn
    X = torch.cat([u, dot_u, double_dot_u], dim=1).cpu().detach().numpy()
    y = np.zeros(X.shape[0])  # Convert your ODE residuals (targets) to a numpy array as well

    # It's a good practice to scale your features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    #
    # # Split data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=42)
    #
    # # Initialize Lasso with cross-validation to find the best alpha
    # lasso = LassoCV(cv=5, random_state=0)
    #
    # # Fit the model to the data
    # lasso.fit(X_train, y_train)

    # # The model coefficients are your estimated parameters
    # estimated_parameters = lasso.coef_
    #
    # # Predict on the test set and compute the mean squared error to evaluate how well the ODE is satisfied
    # y_pred = lasso.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")

    for k in range(u.shape[1]):
        term = hat_m[k] * double_dot_u[:, k] + hat_d[k] * dot_u[:, k]
        for j in range(u.shape[1]):
            term += hat_a[k][j] * torch.sin(u[:, k] - u[:, j])
        term -= hat_P[k]
        ode += term

    return ode


# Define the parameters in matrix form
hat_m_truth = [0.3, 0.2, 0, 0]
hat_d_truth = [0.15, 0.3, 0.25, 0.25]
hat_a_truth = [[0, 1, 0.5, 1.2],
               [1, 0, 1.4, 0.8],
               [0.5, 1.4, 0, 0.1],
               [1.2, 0.8, 0.1, 0]]
hat_P_truth = [0.1, 0.2, -0.1, -0.2]

iterations = 2000
previous_validation_loss = 99999999.0
learning_rate = 1e-2

filename = 'C:/Users/mazhi/PINN_Folder/4_bus_60Hz_10s_trimmed.csv'
df = pd.read_csv(filename, header=None)

time = df.iloc[:, 0].values.reshape(-1, 1)
delta_pool = df.iloc[:, 1:].values

num_of_point = 20
Skip_Point = 10
u_bc, t_bc = evenly_select_data(delta_pool, time, Skip_Point)

# Move tensors to GPU
t_bc = torch.from_numpy(t_bc).float().to(device)
u_bc = torch.from_numpy(u_bc).float().to(device)
time = torch.from_numpy(time).float().to(device)
# delta_pool = torch.from_numpy(delta_pool).float().to(device)

# Calculate the first derivative manually
resolution = 1 / (Skip_Point + 1)
first_derivative_manual = torch.zeros_like(u_bc)
for j in range(u_bc.shape[1]):
    for i in range(1, u_bc.shape[0] - 1):
        first_derivative_manual[i, j] = (u_bc[i + 1, j] - u_bc[i - 1, j]) / (2 * resolution)

# Calculate the second derivative manually
second_derivative_manual = torch.zeros_like(u_bc)
for j in range(u_bc.shape[1]):
    for i in range(1, u_bc.shape[0] - 1):
        second_derivative_manual[i, j] = (
            u_bc[i + 1, j] - 2 * u_bc[i, j] + u_bc[i - 1, j]) / (resolution ** 2)

# # Lasso block
# X = torch.cat((second_derivative_manual, first_derivative_manual, u_bc), dim=1)
# y = torch.zeros(u_bc.shape[0])
#
# # Define and train the Lasso regression model
# model = Lasso(alpha=1.0)  # You can adjust the value of alpha as needed
# model.fit(X, y)
#
# # Extract the estimated coefficients
# params_estimated = model.coef_

# NN network estimation
u = u_bc
dot_u = first_derivative_manual
double_dot_u = second_derivative_manual
t = t_bc

# Variables and their shapes
hat_m = hat_m_truth  # shape: (4,)
hat_d = hat_d_truth # shape: (4,)
hat_P = hat_P_truth  # shape: (4,)
hat_a = hat_a_truth  # shape: (4, 4)
# hat_m = nn.Parameter(torch.randn(4)).to(device)  # shape: (4,)
# hat_d = nn.Parameter(torch.randn(4)).to(device)  # shape: (4,)
# hat_P = nn.Parameter(torch.randn(4)).to(device)  # shape: (4,)
# hat_a = nn.Parameter(torch.randn(4, 4)).to(device)  # shape: (4, 4)

# # Loss function
# loss_fn = nn.MSELoss()
#
# # Optimizer
# optimizer = optim.SGD([hat_m, hat_d, hat_P, hat_a], lr=1e-3)
#
# # Training loop
# num_epochs = 1000
#
# for epoch in range(num_epochs):
#     # Clear the gradients
#     optimizer.zero_grad()
#
#     # Compute the ODE equation
#     ode = torch.zeros_like(u)
#
#     for k in range(u.shape[1]):
#         term = hat_m[k] * double_dot_u[:, k] + hat_d[k] * dot_u[:, k]
#         for j in range(u.shape[1]):
#             term += hat_a[k][j] * torch.sin(u[:, k] - u[:, j])
#         term -= hat_P[k]
#         ode[:, k] = term
#
#     # Compute the loss
#     loss = loss_fn(ode, torch.zeros_like(ode))
#
#     # Perform backpropagation
#     loss.backward()
#
#     # Update the coefficients
#     optimizer.step()
#
#     # Print the loss for monitoring
#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
#
# # Print the estimated coefficients
# print("Estimated Coefficients:")
# print("hat_m:", hat_m.cpu().detach().numpy())
# print("hat_d:", hat_d.cpu().detach().numpy())
# print("hat_P:", hat_P.cpu().detach().numpy())
# print("hat_a:", hat_a.cpu().detach().numpy())

RLC_net = Train_PINN_model(10000, t_bc, u_bc, learning_rate, time)
t_collocation_tensor = time.requires_grad_(True).to(device)
angular = RLC_net(t_collocation_tensor)

angular_np = angular.cpu().detach().numpy()

plt.figure()
color_map = cm.get_cmap('tab10')
time_np = time.cpu().detach().numpy()
u_bc_np, t_bc_np = evenly_select_data(delta_pool, time, Skip_Point)
shift_constant = 1

# Shifting both true and predicted values
shifted_true_values = delta_pool + shift_constant
shifted_predicted_values = angular_np + shift_constant

# Calculating MAPE with shifted values
mape = calculate_mape(shifted_true_values,
                      shifted_predicted_values,
                      threshold=1e-3)
# mape_1 = calculate_mape(delta_pool,angular_np, threshold=0.0005)

# Custom color for the line and the point
line_color = '#FF0000'
point_color = '#FF9E2A'
LR_color_blue = "#00B0F0"
Dotted_line_dark = "#262A2E"
# Custom line width and point size
line_width = 4  # You can set this to your desired value
point_size = 90  # You can set this to your desired value
# Iterate over the columns of angular_np and plot the data
for i in range(angular_np.shape[1]):
    if i <=1:
        plt.plot(time_np, delta_pool[:, i], '-', label=f"Bus {i + 1} truth", color=line_color, linewidth=line_width, zorder=1)
    if i > 1:
        plt.plot(time_np, delta_pool[:, i], '-', label=f"Bus {i + 1} truth", color=LR_color_blue, linewidth=line_width, zorder=1)
        plt.plot(time_np, angular_np[:, i], '--', label=f"Bus {i + 1} predict", color=Dotted_line_dark, linewidth=0.5* line_width, zorder=2)

for i in range(angular_np.shape[1]):
    if i > 1:
        plt.scatter(t_bc_np.cpu().detach().numpy(), u_bc_np[:, i], color=point_color, s=point_size, zorder=3)

# Set the x-axis and y-axis labels


plt.xlabel("Time")
plt.ylabel("Voltage Angle")
# plt.legend()
# plt.title(f'delta Vs delta truth with {Skip_Point + 1} data points before structural MAPE {mape_1}')
plt.show()
#
# for i in range(10):
#     u_bc = angular_np
#     first_derivative_manual, second_derivative_manual = Find_derivative(60, u_bc) # 60Hz
#     # first iteration
#     u_bc = torch.from_numpy(u_bc).float().requires_grad_(True).to(device)
#     RLC_net = Train_PINN_model(2000, time, u_bc, 1e-5, time)
#     anguler = RLC_net(t_collocation_tensor)  # the dependent variable u is given by the network based on independent variables x,t
#     angular_np = anguler.cpu().detach().numpy()
#
# plt.figure()
# color_map = cm.get_cmap('tab10')
# time = time.cpu().detach().numpy()
# u_bc, t_bc = evenly_select_data(delta_pool, time, Skip_Point)
#
# for i in range(angular_np.shape[1]):
#     color = color_map(i)
#     plt.plot(time, angular_np[:, i], '--', label=f"Bus {i + 1} predict", color=color)
#     plt.plot(time, delta_pool[:, i], '-', label=f"Bus {i + 1} truth", color=color)
#     plt.scatter(t_bc, u_bc[:, i], color=color)
#
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.legend()
# plt.title(f'delta Vs delta truth with {Skip_Point + 1} data points')
# plt.show()

