# - 데이터를 읽기 위한 라이브러리
import pandas as pd

# - 수치 계산 라이브러리
import numpy as np

# - 데이터 가공 라이브러리
from sklearn.model_selection import train_test_split

### 평가모델 라이브러리 정의
from sklearn.linear_model import LinearRegression # 선형회귀

# 변환기 모델(클래스) 라이브러리 정의
from sklearn.preprocessing import PolynomialFeatures

### 평가 라이브러리 정의
# 평균절대오차(MAE)
from sklearn.metrics import mean_absolute_error
# 평균제곱오차(MSE)
from sklearn.metrics import mean_squared_error
# 결정계수(R2-score)
from sklearn.metrics import r2_score

# - 데이터 스케일링 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# 클래스 정의
class LinearModel:
    def __init__(self):
        self.linear_results = []  # 결과를 저장할 리스트
    
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
    
    ### 훈련 모델 평가 및 데이터 프레임 저장 함수
    def evaluate_model(self, model_nm, train_score, test_score, 
                       train_mae, test_mae, train_mse, test_mse, train_r2, test_r2):
        
        linear_results = {"model_nm": model_nm}
        
        if train_score < 1 and test_score < 1 and train_score - test_score < 0.09 and train_score - test_score > 0.01:
            
            # result 데이터 프레임에 저장
            linear_results["train_mae"] = train_mae
            linear_results["train_mse"] = train_mse
            linear_results["train_r2"] = train_r2
            linear_results["test_mae"] = test_mae
            linear_results["test_mse"] = test_mse
            linear_results["test_r2"] = test_r2
            linear_results["train_r2-val_r2"] = train_score - test_score
            
            print(f"훈련 결정계수 : {train_score:.4f}, 테스트 결정계수 : {test_score:.4f}, 과적합여부 : {train_score - test_score:.4f}")
            print("해당 모델은 사용 가능한 모델입니다.")
            print(" ")
            print("훈련 데이터 평가")
            print(f"평균절대오차(MAE) : {train_mae:.4f}, 평균제곱오차(MSE) : {train_mse:.4f}, R2_Score : {train_r2:.4f}")
            print("테스트 데이터 평가")
            print(f"평균절대오차(MAE) : {test_mae:.4f}, 평균제곱오차(MSE) : {test_mse:.4f}, R2_Score : {test_r2:.4f}")
            print(" ")
        else:
            print("해당 모델은 사용할 수 없는 모델입니다.")
            print(" ")
            
            # else 조건에 해당하면 나머지 컬럼에 None을 할당
            linear_results["train_mae"] = None
            linear_results["train_mse"] = None
            linear_results["train_r2"] = None
            linear_results["test_mae"] = None
            linear_results["test_mse"] = None
            linear_results["test_r2"] = None
            linear_results["train_r2-val_r2"] = None

        # 결과를 저장
        self.linear_results.append(linear_results)

    ### 저장된 데이터 프레임 호출 함수
    def get_linear_results(self):
        # 결과들을 DataFrame으로 반환
        return pd.DataFrame(self.linear_results)

    ### 선형 회귀 모델 훈련
    def linear_model(self, train_input, train_target, test_input, test_target, model_nm="LinearModel"):
        md = LinearRegression()
        md.fit(train_input, train_target)
        train_score = md.score(train_input, train_target)
        test_score = md.score(test_input, test_target)
        
        train_pred = md.predict(train_input)
        test_pred = md.predict(test_input)
        
        train_mae = mean_absolute_error(train_target, train_pred)
        test_mae = mean_absolute_error(test_target, test_pred)

        train_mse = mean_squared_error(train_target, train_pred)
        test_mse = mean_squared_error(test_target, test_pred)

        train_r2 = r2_score(train_target, train_pred)
        test_r2 = r2_score(test_target, test_pred)        
              
        self.evaluate_model(model_nm, train_score, test_score, 
                            train_mae, test_mae, train_mse, test_mse, train_r2, test_r2)

    ### 폴리노미얼 회귀 모델
    def linear_poly_model(self, train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyModel"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)

        print(f"{degree} 차원 모델이 생성되었습니다.")
        
        self.linear_model(train_poly, train_target, test_poly, test_target, model_nm=f"{model_nm}{degree}")
    
    ### 표준화 스케일링 모델
    def linear_standard_scaled_model(self, train_input, train_target, test_input, test_target, model_nm="LinearStandardModel"):
        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)

        self.linear_model(train_scaled, train_target, test_scaled, test_target, model_nm=model_nm)

    ### MinMax 스케일링 모델
    def linear_minmax_scaled_model(self, train_input, train_target, test_input, test_target, model_nm="LinearMinMaxModel"):
        ss = MinMaxScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)

        self.linear_model(train_scaled, train_target, test_scaled, test_target, model_nm=model_nm)

    ### Robust 스케일링 모델
    def linear_robust_scaled_model(self, train_input, train_target, test_input, test_target, model_nm="LinearRobustModel"):
        ss = RobustScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)

        self.linear_model(train_scaled, train_target, test_scaled, test_target, model_nm=model_nm)
    
    ### 폴리 + 표준화 스케일링 모델
    def linear_poly_standard_model(self, train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyStandardModel"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)

        ss = StandardScaler()
        ss.fit(train_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)

        print(f"{degree} 차원 모델이 생성되었습니다.")
        self.linear_model(train_scaled, train_target, test_scaled, test_target, model_nm=f"{model_nm}{degree}")
    
    ### 폴리 + MinMax 스케일링 모델
    def linear_poly_minmax_model(self, train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyMinMaxModel"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        
        ss = MinMaxScaler()
        ss.fit(train_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)

        print(f"{degree} 차원 모델이 생성되었습니다.")
        self.linear_model(train_scaled, train_target, test_scaled, test_target, model_nm=f"{model_nm}{degree}")
                
    ### 폴리 + Robust 스케일링 모델 
    def linear_poly_robust_model(self, train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyRobustModel"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        
        ss = RobustScaler()
        ss.fit(train_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)
        
        print(f"{degree} 차원 모델이 생성되었습니다.")
        self.linear_model(train_scaled, train_target, test_scaled, test_target, model_nm=f"{model_nm}{degree}")
    

    ### 모든 모델을 비교하는 함수
    def linear_total_model(self, train_input, train_target, test_input, test_target): 
        # 선형 회귀 모델
        print(" -** 선형 회귀 모델 **- ")
        self.linear_model(train_input, train_target, test_input, test_target, model_nm="LinearModel")

        # 선형 + 표준화 스케일링
        print(" -** 선형 + 표준화 스케일링 **- ")
        self.linear_standard_scaled_model(train_input, train_target, test_input, test_target, model_nm="LinearStandardModel")

        # 선형 + MinMax 스케일링
        print(" -** 선형 + MinMax 스케일링 **- ")
        self.linear_minmax_scaled_model(train_input, train_target, test_input, test_target, model_nm="LinearMinMaxModel")

        # 선형 + Robust 스케일링
        print(" -** 선형 + Robust 스케일링 **- ")
        self.linear_robust_scaled_model(train_input, train_target, test_input, test_target, model_nm="LinearRobustModel")

        # 폴리노미얼 회귀 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f"-** 폴리노미얼 {degree} 차 모델 **- ")
            self.linear_poly_model(train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyModel")

        # 폴리 + 표준화 스케일링 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f" -** 폴리노미얼 {degree} 차 + 표준화 스케일링 모델 **- ")
            self.linear_poly_standard_model(train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyStandardModel")

        # 폴리 + MinMax 스케일링 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f" -** 폴리노미얼 {degree} 차 + MinMax 스케일링 모델 **- ")
            self.linear_poly_minmax_model(train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyMinMaxModel")

        # 폴리 + Robust 스케일링 모델 (2, 3, 4 차)
        for degree in range(2, 5):
            print(f" -** 폴리노미얼 {degree} 차 + Robust 스케일링 모델 **- ")
            self.linear_poly_robust_model(train_input, train_target, test_input, test_target, degree, model_nm="LinearPolyRobustModel")