# - 데이터를 읽기 위한 라이브러리
import pandas as pd

# - 수치 계산 라이브러리
import numpy as np

# - 데이터 가공 라이브러리
from sklearn.model_selection import train_test_split

# 선형회귀모델 라이브러리
from sklearn.linear_model import LinearRegression

# 변환기 모델(클래스) 라이브러리 정의
from sklearn.preprocessing import PolynomialFeatures

# - MAE 라이브러리 정의
from sklearn.metrics import mean_absolute_error

# - 데이터 스케일링 라이브러리 정의
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# 클래스 정의
class Model_Util:
    def __init__(self):
        pass
    
    ### 데이터 읽기 및 numpy 배열로 변환
    def data_read(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            self.df.info()
            self.df.head(1)
            read_data = self.df.to_numpy()
            return read_data
        except:
            print("파일 형식이 올바르지 않습니다.")
            return ""
    
    ### 데이터 분할 함수
    def data_split(self, data_full, data_input, test_size):
        train_input, test_input, train_target, test_target = train_test_split(
            data_full, data_input, test_size=test_size, random_state=42)
        return train_input, test_input, train_target, test_target

    ### 훈련 모델 평가 함수
    def evaluate_model(self, train_score, test_score) :
        if train_score < 1 and test_score < 1 and train_score - test_score < 0.09 and  train_score - test_score > 0.01:
            print(f"훈련 결정계수 : {train_score}, 테스트 결정계수 : {test_score}, 과적합여부 : {train_score - test_score}")
            print("해당 모델은 사용 가능한 모델입니다.")
            print(" ")
        else :
            print("해당 모델은 사용할 수 없는 모델입니다.")
            print(" ")

    ### 기본 선형 회귀 모델 훈련
    def new_model(self, train_input, train_target, test_input, test_target):
        lr = LinearRegression()
        lr.fit(train_input, train_target)
        train_score = lr.score(train_input, train_target)
        test_score = lr.score(test_input, test_target)
        self.evaluate_model(train_score, test_score)

    ### 폴리노미얼 회귀 모델
    def poly_model(self, train_input, train_target, test_input, test_target, degree):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)

        print(f"{degree} 차원 모델이 생성되었습니다.")
        self.new_model(train_poly, train_target, test_poly, test_target)
    
    ### 표준화 스케일링 모델
    def standard_scaled_model(self, train_input, train_target, test_input, test_target):
        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)

        self.new_model(train_scaled, train_target, test_scaled, test_target)

    ### MinMax 스케일링 모델
    def minmax_scaled_model(self, train_input, train_target, test_input, test_target):
        ss = MinMaxScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)

        self.new_model(train_scaled, train_target, test_scaled, test_target)

    ### Robust 스케일링 모델
    def robust_scaled_model(self, train_input, train_target, test_input, test_target):
        ss = RobustScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)

        self.new_model(train_scaled, train_target, test_scaled, test_target)
    
    ### 폴리 + 표준화 스케일링 모델
    def poly_standard_model(self, train_input, train_target, test_input, test_target, degree):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)

        ss = StandardScaler()
        ss.fit(train_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)

        print(f"{degree} 차원 모델이 생성되었습니다.")
        self.new_model(train_scaled, train_target, test_scaled, test_target)
    
    ### 폴리 + MinMax 스케일링 모델
    def poly_minmax_model(self, train_input, train_target, test_input, test_target, degree):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        
        ss = MinMaxScaler()
        ss.fit(test_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)

        print(f"{degree} 차원 모델이 생성되었습니다.")
        self.new_model(train_scaled, train_target, test_scaled, test_target)
                
    ### 폴리 + Robust 스케일링 모델 
    def poly_robust_model(self, train_input, train_target, test_input, test_target, degree):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        
        ss = RobustScaler()
        ss.fit(test_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)
        
        print(f"{degree} 차원 모델이 생성되었습니다.")
        self.new_model(train_scaled, train_target, test_scaled, test_target)    


    ### 모든 모델을 비교하는 함수
    def total_model(self, train_input, train_target, test_input, test_target): 
        # 기본 모델
        print("기본 모델")
        self.new_model(train_input, train_target, test_input, test_target)

        # 기본 + 표준화 스케일링
        print("기본 + 표준화 스케일링")
        self.standard_scaled_model(train_input, train_target, test_input, test_target)

        # 기본 + MinMax 스케일링
        print("기본 + MinMax 스케일링")
        self.minmax_scaled_model(train_input, train_target, test_input, test_target)

        # 기본 + Robust 스케일링
        print("기본 + Robust 스케일링")
        self.robust_scaled_model(train_input, train_target, test_input, test_target)

        # 폴리노미얼 회귀 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f"폴리노미얼 {degree} 차 모델")
            self.poly_model(train_input, train_target, test_input, test_target, degree)

        # 폴리 + 표준화 스케일링 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f"폴리노미얼 {degree} 차 + 표준화 스케일링 모델")
            self.poly_standard_model(train_input, train_target, test_input, test_target, degree)

        # 폴리 + MinMax 스케일링 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f"폴리노미얼 {degree} 차 + MinMax 스케일링 모델")
            self.poly_minmax_model(train_input, train_target, test_input, test_target, degree)

        # 폴리 + Robust 스케일링 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f"폴리노미얼 {degree} 차 + Robust 스케일링 모델")
            self.poly_robust_model(train_input, train_target, test_input, test_target, degree)