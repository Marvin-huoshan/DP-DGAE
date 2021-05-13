import pandas as pd
import xlrd

def isLaplace(file):
    df = pd.read_excel(file,engine='openpyxl')
    cont = 0
    data = df.values
    data = [a for b in data for a in b]
    A1 = len([x for x in data if (x > -1 and x <= -0.6)])
    A2 = len([x for x in data if (x > -0.6 and x <= -0.4)])
    A3 = len([x for x in data if (x > -0.4 and x <= -0.2)])
    A4 = len([x for x in data if (x > -0.2 and x <= 0)])
    A5 = len([x for x in data if (x > 0 and x <= 0.2)])
    A6 = len([x for x in data if (x > 0.2 and x <= 0.4)])
    A7 = len([x for x in data if (x > 0.4 and x <= 0.6)])
    A8 = len([x for x in data if (x > 0.6 and x <= 1)])
    A9 = len([x for x in data if (x <= -1)])
    A10 = len([x for x in data if (x > 1)])
    number = 2708*2708
    #sum = A1**2/9.04 + A2**2/6.07 + A3**2/7.42 + A4**2/9.06 + A5**2/9.06 +A6**2/7.42 + A7**2/6.07 + A8**2/9.04 + A9**2/18.39 + A10**2/18.39
    sum = A1 ** 2 / (0.0904*number) + A2 ** 2 / (0.0607*number) + A3 ** 2 / (0.0742*number) + A4 ** 2 / (0.0906*number) + A5 ** 2 / (0.0906*number) + A6 ** 2 / (0.0742*number) + A7 ** 2 / (0.0607*number) + A8 ** 2 / (0.0904*number) + A9 ** 2 / (0.1839*number) + A10 ** (2 / 0.1839*number)
    #卡方
    xx = sum - 2708*2708
    print(xx)

if __name__ == '__main__':
    isLaplace('save_Excel_VGAE_laplace_origin.xlsx')