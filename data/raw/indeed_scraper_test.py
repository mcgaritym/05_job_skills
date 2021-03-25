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
    
    time_0 = time.time()
    job = job.replace(" ", '+')
    url_ = 'https://www.indeed.com/jobs?q=' + job + '&start='
    counter = 0
    list_ = []
    pages = []

    # define number of pages in range to scrape
    for x in range(0, 100000):
        
        time.sleep(5)
            
        # create new url
        url_1 = url_ + str(counter)   
        counter = counter+10

        # get current time
        time_1 = time.time()
        
        # driver clear cookies, get url
        driver.delete_all_cookies()
        driver.get(url_1)
        
        # driver sleep
        driver.execute_script("window.scrollTo(0, 1000)")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, 2000)")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, 1000000)")
        time.sleep(1)
        response_delay = time.time() - time_1
        time.sleep(5*response_delay) 
        
        # driver get page source and beautiful soup for page html
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        date_today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
        
        # start counter for page numbers
        page_count = soup.find('div', id = 'searchCountPages')
        if page_count != None:
            
            page_count = page_count.text.split()[1]
            
            if page_count in pages:
                break
            else:
                pages.append(page_count)
    
            # find all job blocks on 1st page
            a = soup.find_all(name='div', class_=re.compile('jobsearch-SerpJobCard unifiedRow row'))
            for b in a:
                
                time.sleep(1)
                
                # find all <a> blocks with job_title and href link to 2nd page
                c = b.select('h2 > a')
                for d in c:
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
                a_2 = soup_2.find_all(name='h1', class_ = re.compile('title'))
                if a_2 != None:
                    for title in a_2:
                        job_title = title.text
                        print(job_title)
                else:
                    job_title = 'N/A'
                    print(job_title)
                    
                # find company name
                b_2 = soup_2.find_all(name='div', class_ = re.compile('jobsearch-InlineCompanyRating'))
                if b_2 != None:
                    for comp in b_2:
                        b_2_a = comp.find('a')
                        b_2_b = comp.find('div')               
                        if b_2_a != None:
                            company = b_2_a.text
                            print(company)
                        elif b_2_b != None:
                            company = b_2_b.text
                            print(company)
                        else:
                            company = 'N/A'
                            print(company)
                else:
                    company = 'N/A'
                    print(company)
                    
                # find location
                c_2 = soup_2.find_all(name='div', class_ = re.compile('jobsearch-InlineCompanyRating'))
                if c_2 != None:
                    for loc in c_2: 
                        c_2_a = loc.select_one('div:nth-child(4)')
                        c_2_b = loc.select_one('div:nth-child(3)')
        
                        if c_2_a != None:
                            #for c_2_a_a in c_2_a:
                            location = c_2_a.text
                            print(location)
                        elif c_2_b != None:
                            #for c_2_b_b in c_2_b :   
                            location = c_2_b.text
                            print(location)
                        else:
                            location = 'N/A'
                            print(location)                
                else:
                    location = 'N/A'
                    print(location)
                    
                # find rating
                d_2 = soup_2.find(name='div', class_ = re.compile('Ratings-starsCountWrapper'))
                if d_2 != None:
                    rating = d_2['aria-label']
                    print(rating)
                else:
                    rating = 'N/A'
                    print(rating)
                
                # find job qualifications/background bullet points, create list, append bullet points
                bullets = soup_2.select('div > ul > li')
                bullet_list = []
                if bullets != None:   
                    for bul in bullets:
                        job_text = bul.text
                        job_text = job_text.replace('\n', '')
                        job_text = job_text.strip()
                        bullet_list.append(job_text)
                    print(bullet_list)            
                else:
                    job_text = 'N/A'
                    bullet_list.append(job_text)
                    print(bullet_list)
                    
                # append dictionary to list                           
                list_.append(dict({'job_title': job_title, 
                                    'company': company,
                                    'company_rating': rating,
                                    'location': location,
                                    'job_text': bullet_list,
                                    'date': date_today}))    
                
        else:
            break
            
    # create dataframe based on list            
    df = pd.DataFrame(list_, columns=['job_title', 'company', 'company_rating', 'location', 'job_text', 'date'])
    print(df) 
    
    # print to csv file   
    job = job.replace("+", "_")
    df.to_csv(job.upper() + '_indeed_jobs_' + str(datetime.now().strftime("%Y-%m-%d__%H-%M-%S")) + '.csv', index=False)
    time_2 = time.time()
    
    print('\nRuntime (seconds):  ', time_2 - time_0)
    print('\nJobs scraped: ', len(df))
    
    
#job_scraper_skills('data engineer')
job_scraper_skills('data scientist')
#job_scraper_skills('dashboard')
# job_scraper_skills('tableau')
# job_scraper_skills('power BI')
# job_scraper_skills('powerBI')
# job_scraper_skills('bokeh')
# job_scraper_skills('plotly')
# job_scraper_skills('zoho')
# job_scraper_skills('sisense')
# job_scraper_skills('domo')
# job_scraper_skills('chartio')
# job_scraper_skills('google analytics')
# job_scraper_skills('SAP analytics')
# job_scraper_skills('salesforce einstein')
# job_scraper_skills('IBM cognos')







