#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:41:19 2020

@author: mcgaritym
"""

# import packages
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from datetime import date, datetime
import re
import time

# set driver options and request options
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--headless')
options.add_argument('--incognito')
driver = webdriver.Chrome(ChromeDriverManager().install())
headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.39 Safari/537.36', 
           "Accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'}

# set initial counter
counter = 0

def job_scraper_skills(job):
    
    job = job.replace(" ", '+')
    url = 'https://www.indeed.com/jobs?q=' + job + '&start='
    counter = 0
    list_ = []
    
    for x in range(0, 2):
        
        # create new url
        url_new = url + str(counter)   
        counter = counter+10

        # get link, use bs4 to parse
        time_0 = time.time()
        
        
        # driver get url
        driver.get(url_new)
        
        # driver sleep
        driver.execute_script("window.scrollTo(0, 1000)")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, 2000)")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, 1000000)")
        time.sleep(1)
        response_delay = time.time() - time_0
        time.sleep(5*response_delay) 
        
        # driver get page source and beautiful soup for page html
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        date_today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   
        
        # scrape data based on html using bs4
        
        # find all job blocks on 1st page
        a = soup.find_all(name='div', class_=re.compile('jobsearch-SerpJobCard unifiedRow row'))
        for b in a:
            
            # find all <a> blocks with job_title and href link to 2nd page
            c = b.select('h2 > a')
            for d in c:
                                
                'https://www.indeed.com'
                url_2 = d.get('href')
                url_2 = 'https://www.indeed.com' + url_2 
                #print(url_2)
            
            # driver get url_2
            driver.get(url_2)
            
            # driver sleep
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 1000000)")
            
            # driver get page source and beautiful soup for html
            page_source_2 = driver.page_source
            soup_2 = BeautifulSoup(page_source_2, "html.parser")
            
            # find job title
            a_2 = soup_2.find_all(name='h3', class_ = re.compile('title'))
            for title in a_2:
                job_title2 = title.text
                print(job_title2)
            
            # find company name
            b_2 = soup_2.find_all(name='div', class_ = re.compile('jobsearch-InlineCompanyRating'))
            for company in b_2:
                b_2_a = company.find('a')
                b_2_b = company.find('div')               
                if b_2_a != None:
                    company_2 = b_2_a.text
                    print(company_2)
                elif b_2_b != None:
                    company_2 = b_2_b.text
                    print(company_2)
                else:
                    company_2 = 'N/A'
                    print(company_2)
            
            # find location
            c_2 = soup_2.find_all(name='div', class_ = re.compile('jobsearch-InlineCompanyRating'))
            for location in c_2: 
                c_2_a = location.select('div:nth-of-type(4)')
                c_2_b = location.select('div:nth-of-type(3)')

                if c_2_a != None:
                    for c_2_a_a in c_2_a:
                        location_2 = c_2_a_a.text
                        print(location_2)
                elif c_2_b != None:
                    for c_2_b_b in c_2_b :   
                        location_2 = c_2_b_b.text
                        print(location_2)
                else:
                    company_2 = 'N/A'
                    print(location_2)                

            # find rating
            d_2 = soup_2.find(name='div', class_ = re.compile('Ratings-starsCountWrapper'))
            if d_2 != None:
                rating_2 = d_2['aria-label']
                print(rating_2)
            else:
                rating_2 = 'N/A'
                print(rating_2)
            
            # find job qualifications/background bullet points, create list, append bullet points
            bullets_2 = soup_2.select('div > ul > li')
            bullet_list = []
            if bullets_2 != None:   
                for bullets in bullets_2:
                    job_text = bullets.text
                    bullet_list.append(job_text)
                print(bullet_list)            
            
            # append dictionary to list                           
            list_.append(dict({'job_title': job_title2, 
                                'company': company_2,
                                'company_rating': rating_2,
                                'location': location_2,
                                'job_text': bullet_list,
                                'date': date_today}))           

    # create pandas dataframe based on list            
    df = pd.DataFrame(list_, columns=['job_title', 'company', 'company_rating', 'location', 'job_text', 'date'])
    print(df) 
    
    # print to csv file   
    #print(df)
    job = job.replace("+", "_")
    df.to_csv(job.upper() + 'indeed_jobs_' + str(datetime.now().strftime("%Y-%m-%d__%H-%M-%S")) + '.csv', index=False)

# time_1 = time.time()
job_scraper_skills('data engineer')
# time_2 = time.time()
# print('Runtime (seconds):  ', time_2 - time_1)


