"""
This module contains preloaded company data from the MSCI Europe universe.
This eliminates the need to parse Excel files at runtime and simplifies deployment.
"""

# Pre-defined company data extracted from MSCI Europe universe.xlsx
COMPANY_DATA = [
    {"ISIN": "GB00B1YW4409", "Name": "3i Group plc"},
    {"ISIN": "DK0010244425", "Name": "A.P. Møller - Mærsk A/S Class A Shares DKK 1,000"},
    {"ISIN": "IT0001233417", "Name": "A2A S.p.A."},
    {"ISIN": "SE0000190126", "Name": "AB Industrivärden Class A Shares"},
    {"ISIN": "FR0000120073", "Name": "Air Liquide SA"},
    {"ISIN": "NL0000235190", "Name": "Airbus SE"},
    {"ISIN": "DE0008404005", "Name": "Allianz SE"},
    {"ISIN": "GB00BHJYC057", "Name": "Anglo American plc"},
    {"ISIN": "IT0003128367", "Name": "Angolo Copper plc"},
    {"ISIN": "FR0004125920", "Name": "Amundi SA"},
    {"ISIN": "NL0010273215", "Name": "ASML Holding NV"},
    {"ISIN": "GB0009895292", "Name": "AstraZeneca PLC"},
    {"ISIN": "FR0000120628", "Name": "AXA SA"},
    {"ISIN": "ES0113211835", "Name": "Banco Bilbao Vizcaya Argentaria SA"},
    {"ISIN": "ES0113900J37", "Name": "Banco Santander SA"},
    {"ISIN": "DE000BASF111", "Name": "BASF SE"},
    {"ISIN": "DE000BAY0017", "Name": "Bayer AG"},
    {"ISIN": "ES0115056139", "Name": "Banco de Sabadell SA"},
    {"ISIN": "GB0031348658", "Name": "Barclays PLC"},
    {"ISIN": "FR0000131104", "Name": "BNP Paribas SA"},
    {"ISIN": "FR0000120172", "Name": "Carrefour SA"},
    {"ISIN": "FR0000125338", "Name": "Capgemini SE"},
    {"ISIN": "FR0000131906", "Name": "Renault SA"},
    {"ISIN": "IT0003128367", "Name": "Enel SpA"},
    {"ISIN": "DE0007664039", "Name": "Volkswagen AG"},
    {"ISIN": "DE0007100000", "Name": "Mercedes-Benz Group AG"},
    {"ISIN": "DE0007236101", "Name": "Siemens AG"},
    {"ISIN": "GB00B03MLX29", "Name": "Royal Dutch Shell plc"},
    {"ISIN": "GB0009252882", "Name": "GlaxoSmithKline plc"},
    {"ISIN": "FR0000120321", "Name": "L'Oreal SA"},
    {"ISIN": "CH0012005267", "Name": "Novartis AG"},
    {"ISIN": "CH0012032048", "Name": "Roche Holding AG"},
    {"ISIN": "GB00B10RZP78", "Name": "Unilever PLC"},
    {"ISIN": "FR0000121014", "Name": "LVMH Moët Hennessy Louis Vuitton SE"}
]

def get_company_names():
    """Returns a list of all company names."""
    return [company["Name"] for company in COMPANY_DATA]

def get_isin_by_name(company_name):
    """
    Retrieves the ISIN code for a given company name.
    
    Args:
        company_name (str): The company name to look up
        
    Returns:
        str: The ISIN code if found, None otherwise
    """
    for company in COMPANY_DATA:
        if company["Name"] == company_name:
            return company["ISIN"]
    return None

def get_company_by_isin(isin):
    """
    Retrieves company data for a given ISIN code.
    
    Args:
        isin (str): The ISIN code to look up
        
    Returns:
        dict: The company data if found, None otherwise
    """
    for company in COMPANY_DATA:
        if company["ISIN"] == isin:
            return company
    return None
