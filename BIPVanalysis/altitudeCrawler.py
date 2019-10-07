from selenium import webdriver
from bs4 import BeautifulSoup
import os
import re
import random
import time
import pandas as pd
from selenium.webdriver.common.alert import Alert

driver = webdriver.Chrome('./chromedriver')
driver.implicitly_wait(3)

def get_altitude(month) :
    driver.get('https://astro.kasi.re.kr/life/pageView/10?useElevation=1&lat=36.33827134796633&lng=127.39328110552287&elevation=-120.1559744896266&output_range=1&date=2019-'+str(month)+'-15&hour=&minute=&second=&address=%EB%8C%80%EC%A0%84%EA%B4%91%EC%97%AD%EC%8B%9C+%EC%84%9C%EA%B5%AC+%EA%B3%84%EB%A3%A1%EB%A1%9C+%EC%A7%80%ED%95%98+644')

    html = driver.page_source
    soup = BeautifulSoup(html , 'html.parser')
    all_td = soup.select('table > tbody > tr > td')

    month_list = []
    i = 0
    while(i < 24) :
        print(int(str(all_td[i*5])[4:-5]) , "  " , int(str(all_td[2+i*5])[4:-5].split()[0]) )
        month_list.append([month,int(str(all_td[i*5])[4:-5]) , int(str(all_td[2+i*5])[4:-5].split()[0])])
        i = i + 1

    return month_list
def get_yearAltitude() :

    year_list = []
    for month in range(1,13) :
        year_list.extend(get_altitude(month))

    return year_list
def main():
    # print(get_user_list(5))
    # review_list = get_review(2)
    altitude_all_list = get_yearAltitude()
    driver.close()
    print(altitude_all_list)

    altitude_df = pd.DataFrame(altitude_all_list , columns = ['month' , 'hour' , 'altitude'])
    print(altitude_df)
    altitude_df.to_csv("./altitude.csv",  index=False)


if __name__ == "__main__":
    main()
