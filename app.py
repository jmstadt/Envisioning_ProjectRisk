from fastai.tabular.all import *
from flask import Flask, request
import requests
import os.path

path = ''

export_file_url = 'https://www.dropbox.com/s/41vt4mrudfzp49o/Loadspring_Envisioning_ProjectRisk.pkl?dl=1'
export_file_name = 'Loadspring_Envisioning_ProjectRisk.pkl'

def down_load_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)
            
def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        down_load_file(filename, url)
        return True
    return False

download_if_not_exists(export_file_name, export_file_url)

learn = load_learner(export_file_name)

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        
        Application = request.form.get('Application')
        
        ThisPeriodProjectCost = request.form.get('ThisPeriodProjectCost')
        
        PercentComplete = request.form.get('PercentComplete')
        
        EstToCompleteCost = request.form.get('EstToCompleteCost')
        
        LaborCost = request.form.get('LaborCost')
        
        TotalBilled = request.form.get('TotalBilled')
        
        OriginalBudget = request.form.get('OriginalBudget')
        
        PlannedCost = request.form.get('PlannedCost')
        
        ActualCost = request.form.get('ActualCost')
        
        ForecastCost = request.form.get('ForecastCost')
        
        PlannedDuration = request.form.get('PlannedDuration')
        
        ActualDuration = request.form.get('ActualDuration')
        
        RemainingDuration = request.form.get('RemainingDuration')
        
        CostVariance = request.form.get('CostVariance')
        
        ScheduleVariance = request.form.get('ScheduleVariance')
        
        CostPerfIndexByCost = request.form.get('CostPerfIndexByCost')
        
        CostPerfIndexByLaborUnits = request.form.get('CostPerfIndexByLaborUnits')
        
        SchedPerfIndexByCost = request.form.get('SchedPerfIndexByCost')
        
        SchedPerfIndexByLaborUnits = request.form.get('SchedPerfIndexByLaborUnits')
        
        SPI = request.form.get('SPI')
        
        CPI = request.form.get('CPI')
        
        ProfitConfidenceIndexByLaborUnits = request.form.get('ProfitConfidenceIndexByLaborUnits')
        
        ProfitConfidenceIndexByCost = request.form.get('ProfitConfidenceIndexByCost')
        
        EPSId = request.form.get('EPSId')
        
        
        
        
        inf_df = pd.DataFrame(columns=['Application', 'ThisPeriodProjectCost', 'PercentComplete', 
                                       'EstToCompleteCost', 'LaborCost', 'TotalBilled', 'OriginalBudget',
                                       'PlannedCost', 'ActualCost', 'ForecastCost', 'PlannedDuration',
                                       'ActualDuration', 'RemainingDuration', 'CostVariance',
                                       'ScheduleVariance', 'CostPerfIndexByCost', 'CostPerfIndexByLaborUnits',
                                       'SchedPerfIndexByCost', 'SchedPerfIndexByLaborUnits', 'SPI', 'CPI',
                                       'ProfitConfidenceIndexByLaborUnits', 'ProfitConfidenceIndexByCost',
                                       'EPSId'])
        inf_df.loc[0] = [Application, ThisPeriodProjectCost, PercentComplete,
                         EstToCompleteCost, LaborCost, TotalBilled, OriginalBudget,
                         PlannedCost, ActualCost, ForecastCost, PlannedDuration,
                         ActualDuration, RemainingDuration, CostVariance,
                         ScheduleVariance, CostPerfIndexByCost, CostPerfIndexByLaborUnits,
                         SchedPerfIndexByCost, SchedPerfIndexByLaborUnits, SPI, CPI,
                         ProfitConfidenceIndexByLaborUnits, ProfitConfidenceIndexByCost, EPSId]
        
        
        inf_df['ThisPeriodProjectCost'] = pd.to_numeric(inf_df['ThisPeriodProjectCost'], errors='coerce')
        inf_df['PercentComplete'] = pd.to_numeric(inf_df['PercentComplete'], errors='coerce')
        inf_df['EstToCompleteCost'] = pd.to_numeric(inf_df['EstToCompleteCost'], errors='coerce')
        inf_df['LaborCost'] = pd.to_numeric(inf_df['LaborCost'], errors='coerce')
        inf_df['TotalBilled'] = pd.to_numeric(inf_df['TotalBilled'], errors='coerce')
        inf_df['OriginalBudget'] = pd.to_numeric(inf_df['OriginalBudget'], errors='coerce')
        inf_df['PlannedCost'] = pd.to_numeric(inf_df['PlannedCost'], errors='coerce')
        inf_df['ActualCost'] = pd.to_numeric(inf_df['ActualCost'], errors='coerce')
        inf_df['ForecastCost'] = pd.to_numeric(inf_df['ForecastCost'], errors='coerce')
        inf_df['PlannedDuration'] = pd.to_numeric(inf_df['PlannedDuration'], errors='coerce')
        inf_df['ActualDuration'] = pd.to_numeric(inf_df['ActualDuration'], errors='coerce')
        inf_df['RemainingDuration'] = pd.to_numeric(inf_df['RemainingDuration'], errors='coerce')
        inf_df['CostVariance'] = pd.to_numeric(inf_df['CostVariance'], errors='coerce')
        inf_df['ScheduleVariance'] = pd.to_numeric(inf_df['ScheduleVariance'], errors='coerce')
        inf_df['CostPerfIndexByCost'] = pd.to_numeric(inf_df['CostPerfIndexByCost'], errors='coerce')
        inf_df['CostPerfIndexByLaborUnits'] = pd.to_numeric(inf_df['CostPerfIndexByLaborUnits'], errors='coerce')
        inf_df['SchedPerfIndexByCost'] = pd.to_numeric(inf_df['SchedPerfIndexByCost'], errors='coerce')
        inf_df['SchedPerfIndexByLaborUnits'] = pd.to_numeric(inf_df['SchedPerfIndexByLaborUnits'], errors='coerce')
        inf_df['SPI'] = pd.to_numeric(inf_df['SPI'], errors='coerce')
        inf_df['CPI'] = pd.to_numeric(inf_df['CPI'], errors='coerce')
        inf_df['ProfitConfidenceIndexByLaborUnits'] = pd.to_numeric(inf_df['ProfitConfidenceIndexByLaborUnits'], errors='coerce')
        inf_df['ProfitConfidenceIndexByCost'] = pd.to_numeric(inf_df['ProfitConfidenceIndexByCost'], errors='coerce')
        
        inf_row = inf_df.iloc[0]
        
        
        
        pred = learn.predict(inf_row)[1]
        pred_prob = learn.predict(inf_row)[2]
        
        
        return '''The input Application is: {}<br>
                    The input ThisPeriodProjectCost is: {}<br>
                    The input PercentComplete is: {}<br>
                    The input EstToCompleteCost is: {}<br>
                    The input LaborCost is: {}<br>
                    The input TotalBilled is: {}<br>
                    The input OriginalBudget is: {}<br>
                    The input PlannedCost is: {}<br>
                    The input ActualCost is: {}<br>
                    The input ForecastCost is: {}<br>
                    The input PlannedDuration is: {}<br>
                    The input ActualDuration is: {}<br>
                    The input RemainingDuration is: {}<br>
                    The input CostVariance is: {}<br>
                    The input ScheduleVariance is: {}<br>
                    The input CostPerfIndexByCost is: {}<br>
                    The input CostPerfIndexByLaborUnits is: {}<br>
                    The input SchedPerfIndexByCost is: {}<br>
                    The input SchedPerfIndexByLaborUnits is: {}<br>
                    The input SPI is: {}<br>
                    The input CPI is: {}<br>
                    The input ProfitConfidenceIndexByLaborUnits is: {}<br>
                    The input ProfitConfidenceIndexByCost is: {}<br>
                    The input EPSId is: {}<br>
                    <h1>A Prediction of 0 means Late, a prediction of 1 means OnTime. The schedule risk is predicted to be: {}</h1>
                    The confidence probability of Late is the first number. The confidence probability of Ontime is the second number: {}
                    '''.format(Application, ThisPeriodProjectCost, PercentComplete, EstToCompleteCost,
                               LaborCost, TotalBilled, OriginalBudget, PlannedCost, ActualCost, 
                               ForecastCost, PlannedDuration, ActualDuration, RemainingDuration, 
                               CostVariance, ScheduleVariance, CostPerfIndexByCost, CostPerfIndexByLaborUnits,
                               SchedPerfIndexByCost, SchedPerfIndexByLaborUnits, SPI, CPI,
                               ProfitConfidenceIndexByLaborUnits, ProfitConfidenceIndexByCost, EPSId, pred, pred_prob)


    return '''<form method="POST">
                  <h1>Predicting if a project is at risk of being late</h1>
                  
                  Select Application: <select name="Application">
                  <option value="Ares Prism G2 selected">Ares Prism G2</option>
                  <option value="Primavera P6">Primavera P6</option>
                  </select><br>
                  
                  ThisPeriodProjectCost: <input type="number" name="ThisPeriodProjectCost" step=0.01 min=0><br>
                  
                  PercentComplete: <input type="number" name="PercentComplete" step=0.01 min=0 max=100><br>
                  
                  EstToCompleteCost: <input type="number" name="EstToCompleteCost" step=0.01 min=0><br>
                  
                  LaborCost: <input type="number" name="LaborCost" step=0.01 min=0><br>
                  
                  TotalBilled: <input type="number" name="TotalBilled" value=428279.9 step=0.01 min=0><br>
                  
                  OriginalBudget: <input type="number" name="OriginalBudget" step=0.01 min=0><br>
                  
                  PlannedCost: <input type="number" name="PlannedCost" value=5663816 step=0.01 min=0><br>
                  
                  ActualCost: <input type="number" name="ActualCost" value=611754.9 step=0.01 min=0><br>
                  
                  ForecastCost: <input type="number" name="ForecastCost" value=5718291 step=0.01 min=0><br>
                  
                  PlannedDuration: <input type="number" name="PlannedDuration" step=0.01 min=0><br>
                  
                  ActualDuration: <input type="number" name="ActualDuration" step=0.01 min=0><br>
                  
                  RemainingDuration: <input type="number" name="RemainingDuration" step=0.01 min=0><br>
                  
                  CostVariance: <input type="number" name="CostVariance" value=3642449 step=0.01><br>
                  
                  ScheduleVariance: <input type="number" name="ScheduleVariance" value=24 step=0.01 required="required"><br>
                  
                  CostPerfIndexByCost: <input type="number" name="CostPerfIndexByCost" step=0.01 min=0 max=1><br>
                  
                  CostPerfIndexByLaborUnits: <input type="number" name="CostPerfIndexByLaborUnits" step=0.01 min=0 max=1><br>
                  
                  SchedPerfIndexByCost: <input type="number" name="SchedPerfIndexByCost" step=0.01 min=0 max=1><br>
                  
                  SchedPerfIndexByLaborUnits: <input type="number" name="SchedPerfIndexByLaborUnits" step=0.01 min=0 max=1><br>
                  
                  SPI: <input type="number" name="SPI" step=0.01 min=0 max=1><br>
                  
                  CPI: <input type="number" name="CPI" step=0.01 min=0 max=1><br>
                  
                  ProfitConfidenceIndexByLaborUnits: <input type="number" name="ProfitConfidenceIndexByLaborUnits" step=0.01 min=0 max=1><br>
                  
                  ProfitConfidenceIndexByCost: <input type="number" name="ProfitConfidenceIndexByCost" step=0.01 min=0 max=1><br>
                  


                  EPSId: <select name="EPSId">
                  <option value="In-flight">In-flight</option>
                  <option value="Manufacturing">Manufacturing</option>
                  <option value="LOB 2">LOB 2</option>
                  <option value="Energy">Energy</option>
                  <option value="LOB 1">LOB 1</option>
                  <option value="ProdProg1">ProdProg1</option>
                  <option value="E&C">E&C</option>
                  <option value="" select></option>
                  </select><br>
                  
                  <input type="submit" value="Submit"><br>
              </form>'''
