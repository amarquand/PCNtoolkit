#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 10:21:17 2021

@author: seykia
"""
import sys
from subprocess import check_output
import time

def retrieve_jobs():
    
    output = check_output('qstat', shell=True).decode(sys.stdout.encoding)
    output = output.split('\n')
    jobs = dict()
    for line in output[2:-1]:
        (Job_ID, Job_Name, User, Wall_Time, Status, Queue) = line.split()
        jobs[Job_ID] = dict()
        jobs[Job_ID]['name'] = Job_Name
        jobs[Job_ID]['walltime'] = Wall_Time
        jobs[Job_ID]['status'] = Status
        
    return jobs
        

def check_job_status(jobs):
    
    running_jobs = retrieve_jobs()
    
    r = 0
    c = 0
    q = 0
    u = 0
    for job in jobs:
        if running_jobs[job]['status'] == 'C':
            c += 1
        elif running_jobs[job]['status'] == 'Q':
            q += 1
        elif running_jobs[job]['status'] == 'R':
            r += 1
        else:
            u += 1
                 
    print('Total:%d, Queued: %d, Running:%d, Completed:%d, Unknown:%d' 
          %(len(jobs), q, r, c, u))
    return q,r,c,u
    

def check_jobs(jobs, delay=60):
    
    n = len(jobs)
    
    while(True):
        q,r,c,u = check_job_status(jobs)
        if c == n:
            print('All jobs are finished!')
            break
        time.sleep(delay)
        
    
        
        
    
            
        
        