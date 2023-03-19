import customtkinter
import pandas as pd #imports panda library
from sklearn.model_selection import train_test_split # 
import tensorflow as tf
from sklearn.linear_model import LinearRegression

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("600x350")

def predict():
    dataset = pd.read_csv("INDINF_CPI.csv")

    x = dataset.drop(columns=["INDINF_CPI_COMMON_Q"])
    y = dataset["INDINF_CPI_COMMON_Q"]

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1)

    clf = LinearRegression()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Inflation Predictor")

label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Predict", command=predict)

button.pack(pady=12, padx=10)

root.mainloop()