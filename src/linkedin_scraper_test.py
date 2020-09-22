#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import packages
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
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
    url_ = 'https://www.linkedin.com/jobs/search?keywords=' + job + "&f_TP=1" 
    list_ = []

    # get current time
    time_0 = time.time()
    
    # driver clear cookies, get url
    driver.delete_all_cookies()
    driver.get(url_)
    
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
        # Wait to load page
        time.sleep(2)
    
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    # counter = 0
    # while (counter < 50):
            
    #         counter = counter+1
    #         print(counter)
    #         try:
    #             #button_more = WebDriverWait(driver, 5).until(EC.presence_of_element_located(By.CLASS_NAME, "infinite-scroller__show-more-button infinite-scroller__show-more-button--visible"))
    #             time.sleep(2)
    #             button_more = driver.find_element_by_xpath('//*[@id="main-content"]/div/section/button')
    #             button_more.click()
    #         except TimeoutException:
    #             break
    counts = 0         
    while True & (counts < 50):
            counts = counts+1
            try:
                time.sleep(2)
                button_more = driver.find_element_by_xpath('//*[@id="main-content"]/div/section/button')
                button_more.click()
            except:
                break           
            
            
    # pause for 10 seconds
    time.sleep(10)
        
    # driver get page source and beautiful soup for page html
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    date_scraped = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
    
    # find all job blocks
    a = soup.find_all(name='li', class_ = re.compile('result-card job-result-card'))
    for b in a:     
            
        # find all <a> blocks with job_title and href link to 2nd page
        c = b.find(name='a', class_ = 'result-card__full-card-link')
        url_2 = c['href']
        
        # driver get url_2
        driver.get(url_2)
        
        # driver sleep
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, 1000000)")
        
        # driver get page source and beautiful soup for html
        page_source_2 = driver.page_source
        soup_2 = BeautifulSoup(page_source_2, "html.parser")
        
        # find topcard details
        topcard = soup_2.find_all(name='section', class_ = 'topcard')
        for info in topcard:
            
            # JOB TITLE
            info_job_title = info.find(name='h2', class_ = 'topcard__title')
            if info_job_title != None:
                job_title = info_job_title.text
            else:
                job_title = 'N/A'
            
            # COMPANY NAME        
            info_company = info.select_one('h3:nth-child(2)')
            if info_company != None:
                company = info_company.find(name = 'a', class_ = re.compile('topcard__org-name'))
                if company == None:
                    company = 'N/A' 
                elif company.text != None:
                    company = company.text
                else:
                    company = company
            else:
                company = 'N/A'                
            
            # JOB LOCATION
            info_location = info.select_one('h3:nth-child(2)')
            if info_location != None:
                location = info_location.find(name = 'span', class_ = re.compile('topcard__flavor--bullet'))
                if location == None:
                    location = 'N/A'
                elif location.text != None:
                    location = location.text
                else:
                    location = location
            else:
                location = 'N/A'                 
            
            # DATE POSTED
            info_date_posted = info.select_one('h3:nth-child(3)')
            if info_date_posted != None:
                date_posted = info_date_posted.find(name = 'span', class_ = re.compile('posted-time-ago'))
                
                if date_posted != None:
                    date_posted = date_posted.text
                else:
                    date_posted = date_posted

            else:
                date_posted = 'N/A'                  
            
            # NUMBER OF APPLICANTS
            info_applicants = info.select_one('h3:nth-child(3)')
            if info_applicants != None:
                applicants_1 = info_applicants.find(name = 'span', class_ = re.compile('num-applicants__caption'))
                applicants_2 = info_applicants.find(name = 'figcaption', class_ = re.compile('num-applicants__caption'))

                if applicants_1 != None:
                    applicants = applicants_1.text
                elif applicants_2 != None:
                    applicants = applicants_2.text
                else:
                    applicants_2 = 'Unknown'
            else:
                applicants = 'N/A'                  

        # find topcard details
        job_text_list = []
        description = soup_2.find_all(name='div', class_ = 'show-more-less-html__markup')
        if description != None:
            
            for item in description:
                
                description_0 = item
                description_1 = item.select('ul')
                description_2 = item.select('ul > li')
                description_3 = item.select('strong')
                description_4 = item.select('strong > u')
                description_5 = item.select('em')
                description_6 = item.select('u')
                description_7 = item.select('p')
                description_8 = item.select('li')

                if description_0 != None:
                    for item_0 in description_0:
                        if item_0 == None:  
                            continue
                        elif type(item_0) == 'bs4.element.NavigableString':
                            continue
                        elif item_0.text != None:
                            job_text = item_0.text
                            job_text_list.append(job_text)
                        else:
                            pass
                
                if description_1 != None:
                    for item_1 in description_1:
                        if item_1 == None:  
                            continue
                        elif type(item_1) == 'bs4.element.NavigableString':
                            continue
                        elif item_1.text != None:
                            job_text = item_1.text
                            job_text_list.append(job_text)
                        else:
                            pass
                        
                if description_2 != None:
                    for item_2 in description_2:
                        if item_2 == None:  
                            continue
                        elif type(item_2) == 'bs4.element.NavigableString':
                            continue
                        elif item_2.text != None:
                            job_text = item_2.text
                            job_text_list.append(job_text)
                        else:
                            pass   

                if description_3 != None:
                    for item_3 in description_3:
                        if item_3 == None:  
                            continue
                        elif type(item_3) == 'bs4.element.NavigableString':
                            continue
                        elif item_3.text != None:
                            job_text = item_3.text
                            job_text_list.append(job_text)
                        else:
                            pass
                
                if description_4 != None:
                    for item_4 in description_4:
                        if item_4 == None:  
                            continue
                        elif type(item_4) == 'bs4.element.NavigableString':
                            continue
                        elif item_4.text != None:
                            job_text = item_4.text
                            job_text_list.append(job_text)
                        else:
                            pass
                        
                if description_5 != None:
                    for item_5 in description_5:
                        if item_5 == None:  
                            continue
                        elif type(item_5) == 'bs4.element.NavigableString':
                            continue
                        elif item_5.text != None:
                            job_text = item_5.text
                            job_text_list.append(job_text)
                        else:
                            pass   

                if description_6 != None:
                    for item_6 in description_6:
                        if item_6 == None:  
                            continue
                        elif type(item_6) == 'bs4.element.NavigableString':
                            continue
                        elif item_6.text != None:
                            job_text = item_6.text
                            job_text_list.append(job_text)
                        else:
                            pass   

                if description_7 != None:
                    for item_7 in description_7:
                        if item_7 == None:  
                            continue
                        elif type(item_7) == 'bs4.element.NavigableString':
                            continue
                        elif item_7.text != None:
                            job_text = item_7.text
                            job_text_list.append(job_text)
                        else:
                            pass   

                if description_8 != None:
                    for item_8 in description_8:
                        if item_8 == None:  
                            continue
                        elif type(item_8) == 'bs4.element.NavigableString':
                            continue
                        elif item_8.text != None:
                            job_text = item_8.text
                            job_text_list.append(job_text)
                        else:
                            pass   

                
        # find criteria details
        criteria = soup_2.find_all(name='ul', class_ = 'job-criteria__list')
        for info in criteria:
            
            # find seniority level
            info_seniority_level = info.select_one('li:nth-child(1)')
            if info_seniority_level != None:
                seniority_level = info_seniority_level.span.text
            else:
                seniority_level = 'N/A'
            
            # find employment type
            info_employment_type = info.select_one('li:nth-child(2)')
            if info_employment_type != None:
                employment_type = info_employment_type.span.text
            else:
                employment_type = 'N/A'
                
            # find job function
            info_job_function = info.select_one('li:nth-child(3)')
            if info_job_function != None:
                job_function = info_job_function.span.text
            else:
                job_function = 'N/A'  
                            
            # find industries
            info_industries = info.select_one('li:nth-child(4)')
            if info_industries != None:
                industries = info_industries.span.text
            else:
                industries = 'N/A'    
        
        # print statement
        print(job_title)  
        print(company)
        print(location)
        print(date_posted)
        print(applicants)
        print(job_text_list)
        print(seniority_level) 
        print(employment_type) 
        print(job_function) 
        print(industries) 
        print(date_scraped) 
        
        # append dictionary to list
        list_.append(dict({
                'job_title': job_title, 
                'company': company,
                'location': location,
                'date_posted': date_posted,
                'applicants': applicants,
                'job_text': job_text_list, 
                'seniority_level': seniority_level,
                'employment_type': employment_type,
                'job_function': job_function,
                'industries': industries,
                'date_scraped': date_scraped
                }))

    # create pandas dataframe based on list            
    df = pd.DataFrame(list_, columns=['job_title', 'company', 'location', 'date_posted', 'applicants', 'job_text', 'seniority_level', 'employment_type', 'job_function', 'industries','date_scraped'])
    print(df) 
    
    # print to csv file   
    job = job.replace("+", "_")
    df.to_csv(job.upper() + '_linkedin_jobs_' + str(datetime.now().strftime("%Y-%m-%d__%H-%M-%S")) + '.csv', index=False)
    time_1 = time.time()
    
    print('\nRuntime (seconds):  ', time_1 - time_0)
    print('\nJobs scraped: ', len(df))
    
job_scraper_skills('data scientist')
job_scraper_skills('data engineer')







