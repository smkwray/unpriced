from __future__ import annotations

import pandas as pd


def atus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"respondent_id": 1, "state_fips": "06", "year": 2019, "subgroup": "all", "active_household_childcare_hours": 24.0, "active_nonhousehold_childcare_hours": 7.0, "active_childcare_hours": 31.0, "supervisory_childcare_hours": 11.0, "childcare_hours": 31.0, "weight": 1.2, "parent_employment_rate": 0.76, "single_parent_share": 0.22, "median_income": 72000, "unemployment_rate": 0.041, "births": 500000},
            {"respondent_id": 2, "state_fips": "06", "year": 2020, "subgroup": "all", "active_household_childcare_hours": 28.0, "active_nonhousehold_childcare_hours": 8.0, "active_childcare_hours": 36.0, "supervisory_childcare_hours": 13.0, "childcare_hours": 36.0, "weight": 1.1, "parent_employment_rate": 0.69, "single_parent_share": 0.23, "median_income": 73500, "unemployment_rate": 0.092, "births": 485000},
            {"respondent_id": 3, "state_fips": "06", "year": 2021, "subgroup": "all", "active_household_childcare_hours": 22.5, "active_nonhousehold_childcare_hours": 7.0, "active_childcare_hours": 29.5, "supervisory_childcare_hours": 12.0, "childcare_hours": 29.5, "weight": 1.0, "parent_employment_rate": 0.75, "single_parent_share": 0.22, "median_income": 76000, "unemployment_rate": 0.068, "births": 492000},
            {"respondent_id": 4, "state_fips": "48", "year": 2019, "subgroup": "all", "active_household_childcare_hours": 22.0, "active_nonhousehold_childcare_hours": 6.0, "active_childcare_hours": 28.0, "supervisory_childcare_hours": 10.0, "childcare_hours": 28.0, "weight": 1.3, "parent_employment_rate": 0.72, "single_parent_share": 0.24, "median_income": 64000, "unemployment_rate": 0.038, "births": 390000},
            {"respondent_id": 5, "state_fips": "48", "year": 2020, "subgroup": "all", "active_household_childcare_hours": 26.0, "active_nonhousehold_childcare_hours": 7.5, "active_childcare_hours": 33.5, "supervisory_childcare_hours": 12.5, "childcare_hours": 33.5, "weight": 1.2, "parent_employment_rate": 0.66, "single_parent_share": 0.25, "median_income": 65000, "unemployment_rate": 0.081, "births": 381000},
            {"respondent_id": 6, "state_fips": "48", "year": 2021, "subgroup": "all", "active_household_childcare_hours": 20.5, "active_nonhousehold_childcare_hours": 6.5, "active_childcare_hours": 27.0, "supervisory_childcare_hours": 10.5, "childcare_hours": 27.0, "weight": 1.1, "parent_employment_rate": 0.73, "single_parent_share": 0.24, "median_income": 67500, "unemployment_rate": 0.061, "births": 388000},
            {"respondent_id": 7, "state_fips": "36", "year": 2019, "subgroup": "all", "active_household_childcare_hours": 23.5, "active_nonhousehold_childcare_hours": 7.0, "active_childcare_hours": 30.5, "supervisory_childcare_hours": 11.0, "childcare_hours": 30.5, "weight": 1.1, "parent_employment_rate": 0.74, "single_parent_share": 0.21, "median_income": 70000, "unemployment_rate": 0.040, "births": 230000},
            {"respondent_id": 8, "state_fips": "36", "year": 2020, "subgroup": "all", "active_household_childcare_hours": 27.0, "active_nonhousehold_childcare_hours": 8.0, "active_childcare_hours": 35.0, "supervisory_childcare_hours": 12.0, "childcare_hours": 35.0, "weight": 1.0, "parent_employment_rate": 0.67, "single_parent_share": 0.21, "median_income": 70500, "unemployment_rate": 0.096, "births": 220000},
            {"respondent_id": 9, "state_fips": "36", "year": 2021, "subgroup": "all", "active_household_childcare_hours": 22.2, "active_nonhousehold_childcare_hours": 6.6, "active_childcare_hours": 28.8, "supervisory_childcare_hours": 10.8, "childcare_hours": 28.8, "weight": 1.0, "parent_employment_rate": 0.74, "single_parent_share": 0.20, "median_income": 73000, "unemployment_rate": 0.071, "births": 225000},
        ]
    )


def ndcp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"county_fips": "06037", "state_fips": "06", "year": 2019, "child_age": "infant", "provider_type": "center", "annual_price": 15400, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "06037", "state_fips": "06", "year": 2020, "child_age": "infant", "provider_type": "center", "annual_price": 16100, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "06037", "state_fips": "06", "year": 2021, "child_age": "infant", "provider_type": "center", "annual_price": 16550, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "06073", "state_fips": "06", "year": 2019, "child_age": "preschool", "provider_type": "home", "annual_price": 10800, "imputed_flag": 1, "sample_weight": 0.8},
            {"county_fips": "06073", "state_fips": "06", "year": 2020, "child_age": "preschool", "provider_type": "home", "annual_price": 11150, "imputed_flag": 1, "sample_weight": 0.8},
            {"county_fips": "06073", "state_fips": "06", "year": 2021, "child_age": "preschool", "provider_type": "home", "annual_price": 11400, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "48113", "state_fips": "48", "year": 2019, "child_age": "infant", "provider_type": "center", "annual_price": 9800, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "48113", "state_fips": "48", "year": 2020, "child_age": "infant", "provider_type": "center", "annual_price": 10100, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "48113", "state_fips": "48", "year": 2021, "child_age": "infant", "provider_type": "center", "annual_price": 10450, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "48201", "state_fips": "48", "year": 2019, "child_age": "preschool", "provider_type": "home", "annual_price": 7600, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "48201", "state_fips": "48", "year": 2020, "child_age": "preschool", "provider_type": "home", "annual_price": 7900, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "48201", "state_fips": "48", "year": 2021, "child_age": "preschool", "provider_type": "home", "annual_price": 8150, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "36061", "state_fips": "36", "year": 2019, "child_age": "infant", "provider_type": "center", "annual_price": 17800, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "36061", "state_fips": "36", "year": 2020, "child_age": "infant", "provider_type": "center", "annual_price": 18300, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "36061", "state_fips": "36", "year": 2021, "child_age": "infant", "provider_type": "center", "annual_price": 18850, "imputed_flag": 0, "sample_weight": 1.0},
            {"county_fips": "36029", "state_fips": "36", "year": 2019, "child_age": "preschool", "provider_type": "home", "annual_price": 9600, "imputed_flag": 1, "sample_weight": 0.7},
            {"county_fips": "36029", "state_fips": "36", "year": 2020, "child_age": "preschool", "provider_type": "home", "annual_price": 9850, "imputed_flag": 1, "sample_weight": 0.7},
            {"county_fips": "36029", "state_fips": "36", "year": 2021, "child_age": "preschool", "provider_type": "home", "annual_price": 10100, "imputed_flag": 1, "sample_weight": 0.7},
        ]
    )


def qcew() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"county_fips": "06037", "state_fips": "06", "year": 2019, "naics": "624410", "childcare_worker_wage": 16.1, "outside_option_wage": 21.8, "employment": 55000},
            {"county_fips": "06037", "state_fips": "06", "year": 2020, "naics": "624410", "childcare_worker_wage": 16.4, "outside_option_wage": 22.0, "employment": 51000},
            {"county_fips": "06037", "state_fips": "06", "year": 2021, "naics": "624410", "childcare_worker_wage": 16.8, "outside_option_wage": 22.4, "employment": 53000},
            {"county_fips": "06073", "state_fips": "06", "year": 2019, "naics": "624410", "childcare_worker_wage": 15.5, "outside_option_wage": 20.7, "employment": 18000},
            {"county_fips": "06073", "state_fips": "06", "year": 2020, "naics": "624410", "childcare_worker_wage": 15.9, "outside_option_wage": 21.0, "employment": 17000},
            {"county_fips": "06073", "state_fips": "06", "year": 2021, "naics": "624410", "childcare_worker_wage": 16.2, "outside_option_wage": 21.5, "employment": 17500},
            {"county_fips": "48113", "state_fips": "48", "year": 2019, "naics": "624410", "childcare_worker_wage": 12.5, "outside_option_wage": 18.0, "employment": 21000},
            {"county_fips": "48113", "state_fips": "48", "year": 2020, "naics": "624410", "childcare_worker_wage": 12.7, "outside_option_wage": 18.2, "employment": 20200},
            {"county_fips": "48113", "state_fips": "48", "year": 2021, "naics": "624410", "childcare_worker_wage": 13.0, "outside_option_wage": 18.7, "employment": 21500},
            {"county_fips": "48201", "state_fips": "48", "year": 2019, "naics": "624410", "childcare_worker_wage": 11.9, "outside_option_wage": 17.5, "employment": 26000},
            {"county_fips": "48201", "state_fips": "48", "year": 2020, "naics": "624410", "childcare_worker_wage": 12.2, "outside_option_wage": 17.8, "employment": 24900},
            {"county_fips": "48201", "state_fips": "48", "year": 2021, "naics": "624410", "childcare_worker_wage": 12.5, "outside_option_wage": 18.2, "employment": 25500},
            {"county_fips": "36061", "state_fips": "36", "year": 2019, "naics": "624410", "childcare_worker_wage": 17.0, "outside_option_wage": 24.0, "employment": 30000},
            {"county_fips": "36061", "state_fips": "36", "year": 2020, "naics": "624410", "childcare_worker_wage": 17.4, "outside_option_wage": 24.3, "employment": 27900},
            {"county_fips": "36061", "state_fips": "36", "year": 2021, "naics": "624410", "childcare_worker_wage": 17.8, "outside_option_wage": 24.8, "employment": 29100},
            {"county_fips": "36029", "state_fips": "36", "year": 2019, "naics": "624410", "childcare_worker_wage": 13.6, "outside_option_wage": 19.7, "employment": 9200},
            {"county_fips": "36029", "state_fips": "36", "year": 2020, "naics": "624410", "childcare_worker_wage": 13.9, "outside_option_wage": 20.1, "employment": 8900},
            {"county_fips": "36029", "state_fips": "36", "year": 2021, "naics": "624410", "childcare_worker_wage": 14.1, "outside_option_wage": 20.4, "employment": 9100},
        ]
    )


def acs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"county_fips": "06037", "state_fips": "06", "year": 2019, "median_income": 72000, "under5_male_population": 326000, "under5_female_population": 314000, "under5_population": 640000, "rent_index": 1.45, "commuting_minutes": 30.1, "single_parent_share": 0.22, "unemployment_rate": 0.041},
            {"county_fips": "06037", "state_fips": "06", "year": 2020, "median_income": 73500, "under5_male_population": 319000, "under5_female_population": 307000, "under5_population": 626000, "rent_index": 1.48, "commuting_minutes": 29.4, "single_parent_share": 0.23, "unemployment_rate": 0.092},
            {"county_fips": "06037", "state_fips": "06", "year": 2021, "median_income": 76000, "under5_male_population": 316000, "under5_female_population": 304000, "under5_population": 620000, "rent_index": 1.52, "commuting_minutes": 28.8, "single_parent_share": 0.22, "unemployment_rate": 0.068},
            {"county_fips": "06073", "state_fips": "06", "year": 2019, "median_income": 69000, "under5_male_population": 96000, "under5_female_population": 92000, "under5_population": 188000, "rent_index": 1.32, "commuting_minutes": 27.0, "single_parent_share": 0.20, "unemployment_rate": 0.037},
            {"county_fips": "06073", "state_fips": "06", "year": 2020, "median_income": 70500, "under5_male_population": 93500, "under5_female_population": 89500, "under5_population": 183000, "rent_index": 1.34, "commuting_minutes": 25.9, "single_parent_share": 0.20, "unemployment_rate": 0.085},
            {"county_fips": "06073", "state_fips": "06", "year": 2021, "median_income": 72000, "under5_male_population": 92500, "under5_female_population": 88500, "under5_population": 181000, "rent_index": 1.37, "commuting_minutes": 26.2, "single_parent_share": 0.19, "unemployment_rate": 0.061},
            {"county_fips": "48113", "state_fips": "48", "year": 2019, "median_income": 64000, "under5_male_population": 112000, "under5_female_population": 108000, "under5_population": 220000, "rent_index": 1.12, "commuting_minutes": 28.2, "single_parent_share": 0.24, "unemployment_rate": 0.038},
            {"county_fips": "48113", "state_fips": "48", "year": 2020, "median_income": 65000, "under5_male_population": 109500, "under5_female_population": 105500, "under5_population": 215000, "rent_index": 1.13, "commuting_minutes": 27.1, "single_parent_share": 0.25, "unemployment_rate": 0.081},
            {"county_fips": "48113", "state_fips": "48", "year": 2021, "median_income": 67500, "under5_male_population": 107000, "under5_female_population": 103000, "under5_population": 210000, "rent_index": 1.16, "commuting_minutes": 27.4, "single_parent_share": 0.24, "unemployment_rate": 0.061},
            {"county_fips": "48201", "state_fips": "48", "year": 2019, "median_income": 59000, "under5_male_population": 214000, "under5_female_population": 206000, "under5_population": 420000, "rent_index": 1.09, "commuting_minutes": 31.0, "single_parent_share": 0.27, "unemployment_rate": 0.040},
            {"county_fips": "48201", "state_fips": "48", "year": 2020, "median_income": 60200, "under5_male_population": 209000, "under5_female_population": 201000, "under5_population": 410000, "rent_index": 1.11, "commuting_minutes": 29.7, "single_parent_share": 0.28, "unemployment_rate": 0.084},
            {"county_fips": "48201", "state_fips": "48", "year": 2021, "median_income": 61800, "under5_male_population": 206000, "under5_female_population": 199000, "under5_population": 405000, "rent_index": 1.14, "commuting_minutes": 30.1, "single_parent_share": 0.27, "unemployment_rate": 0.064},
            {"county_fips": "36061", "state_fips": "36", "year": 2019, "median_income": 83000, "under5_male_population": 47000, "under5_female_population": 45000, "under5_population": 92000, "rent_index": 1.72, "commuting_minutes": 31.2, "single_parent_share": 0.19, "unemployment_rate": 0.033},
            {"county_fips": "36061", "state_fips": "36", "year": 2020, "median_income": 84500, "under5_male_population": 46500, "under5_female_population": 44500, "under5_population": 91000, "rent_index": 1.76, "commuting_minutes": 29.8, "single_parent_share": 0.19, "unemployment_rate": 0.099},
            {"county_fips": "36061", "state_fips": "36", "year": 2021, "median_income": 87000, "under5_male_population": 46000, "under5_female_population": 44000, "under5_population": 90000, "rent_index": 1.81, "commuting_minutes": 29.2, "single_parent_share": 0.18, "unemployment_rate": 0.073},
            {"county_fips": "36029", "state_fips": "36", "year": 2019, "median_income": 61000, "under5_male_population": 21500, "under5_female_population": 20500, "under5_population": 42000, "rent_index": 1.03, "commuting_minutes": 23.0, "single_parent_share": 0.22, "unemployment_rate": 0.039},
            {"county_fips": "36029", "state_fips": "36", "year": 2020, "median_income": 62000, "under5_male_population": 21000, "under5_female_population": 20000, "under5_population": 41000, "rent_index": 1.04, "commuting_minutes": 22.4, "single_parent_share": 0.22, "unemployment_rate": 0.088},
            {"county_fips": "36029", "state_fips": "36", "year": 2021, "median_income": 64000, "under5_male_population": 20750, "under5_female_population": 19750, "under5_population": 40500, "rent_index": 1.06, "commuting_minutes": 22.7, "single_parent_share": 0.21, "unemployment_rate": 0.067},
        ]
    )


def nes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"county_fips": "06037", "state_fips": "06", "year": 2021, "nonemployer_firms": 1186, "receipts": 4721000.0, "source_url": "https://api.census.gov/data/2021/nonemp"},
            {"county_fips": "06073", "state_fips": "06", "year": 2021, "nonemployer_firms": 422, "receipts": 1710000.0, "source_url": "https://api.census.gov/data/2021/nonemp"},
            {"county_fips": "48113", "state_fips": "48", "year": 2021, "nonemployer_firms": 356, "receipts": 1254000.0, "source_url": "https://api.census.gov/data/2021/nonemp"},
            {"county_fips": "48201", "state_fips": "48", "year": 2021, "nonemployer_firms": 777, "receipts": 3018000.0, "source_url": "https://api.census.gov/data/2021/nonemp"},
            {"county_fips": "36061", "state_fips": "36", "year": 2021, "nonemployer_firms": 664, "receipts": 2842000.0, "source_url": "https://api.census.gov/data/2021/nonemp"},
            {"county_fips": "36029", "state_fips": "36", "year": 2021, "nonemployer_firms": 99, "receipts": 388000.0, "source_url": "https://api.census.gov/data/2021/nonemp"},
        ]
    )


def head_start() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"county_fips": "06037", "state_fips": "06", "head_start_slots": 3120.0, "open_locations": 28},
            {"county_fips": "06073", "state_fips": "06", "head_start_slots": 1140.0, "open_locations": 11},
            {"county_fips": "48113", "state_fips": "48", "head_start_slots": 980.0, "open_locations": 9},
            {"county_fips": "48201", "state_fips": "48", "head_start_slots": 2560.0, "open_locations": 24},
            {"county_fips": "36061", "state_fips": "36", "head_start_slots": 720.0, "open_locations": 8},
            {"county_fips": "36029", "state_fips": "36", "head_start_slots": 260.0, "open_locations": 3},
        ]
    )


def nces_ccd() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": "06", "prek_schools": 820, "kg_schools": 5150, "pk_or_kg_schools": 5968, "operational_schools": 9800, "public_school_option_index": 0.609},
            {"state_fips": "48", "prek_schools": 740, "kg_schools": 6050, "pk_or_kg_schools": 6792, "operational_schools": 9500, "public_school_option_index": 0.715},
            {"state_fips": "36", "prek_schools": 510, "kg_schools": 3650, "pk_or_kg_schools": 4160, "operational_schools": 4700, "public_school_option_index": 0.885},
        ]
    )


def licensing_supply_shocks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2020,
                "center_infant_ratio": 4.0,
                "center_toddler_ratio": 7.0,
                "center_infant_group_size": 8.0,
                "center_toddler_group_size": 14.0,
                "shock_label": "sample_center_ratio_baseline",
                "effective_date": "2020-01-01",
                "source_url": "https://childcareta.acf.hhs.gov/sites/default/files/new-occ/resource/files/center_licensing_trends_brief_2014.pdf",
            },
            {
                "state_fips": "06",
                "year": 2021,
                "center_infant_ratio": 3.0,
                "center_toddler_ratio": 6.0,
                "center_infant_group_size": 6.0,
                "center_toddler_group_size": 12.0,
                "shock_label": "sample_center_ratio_tightening",
                "effective_date": "2021-01-01",
                "source_url": "https://childcareta.acf.hhs.gov/sites/default/files/new-occ/resource/files/center_licensing_trends_brief_2014.pdf",
            },
            {
                "state_fips": "48",
                "year": 2020,
                "center_infant_ratio": 4.0,
                "center_toddler_ratio": 7.0,
                "center_infant_group_size": 8.0,
                "center_toddler_group_size": 14.0,
                "shock_label": "sample_center_ratio_baseline",
                "effective_date": "2020-01-01",
                "source_url": "https://childcareta.acf.hhs.gov/sites/default/files/new-occ/resource/files/center_licensing_trends_brief_2014.pdf",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "center_infant_ratio": 3.0,
                "center_toddler_ratio": 6.0,
                "center_infant_group_size": 6.0,
                "center_toddler_group_size": 12.0,
                "shock_label": "sample_center_ratio_tightening",
                "effective_date": "2021-01-01",
                "source_url": "https://childcareta.acf.hhs.gov/sites/default/files/new-occ/resource/files/center_licensing_trends_brief_2014.pdf",
            },
            {
                "state_fips": "36",
                "year": 2020,
                "center_infant_ratio": 4.0,
                "center_toddler_ratio": 7.0,
                "center_infant_group_size": 8.0,
                "center_toddler_group_size": 14.0,
                "shock_label": "sample_center_ratio_baseline",
                "effective_date": "2020-01-01",
                "source_url": "https://childcareta.acf.hhs.gov/sites/default/files/new-occ/resource/files/center_licensing_trends_brief_2014.pdf",
            },
            {
                "state_fips": "36",
                "year": 2021,
                "center_infant_ratio": 4.0,
                "center_toddler_ratio": 7.0,
                "center_infant_group_size": 8.0,
                "center_toddler_group_size": 14.0,
                "shock_label": "sample_center_ratio_no_change",
                "effective_date": "2021-01-01",
                "source_url": "https://childcareta.acf.hhs.gov/sites/default/files/new-occ/resource/files/center_licensing_trends_brief_2014.pdf",
            },
        ]
    )


def ccdf() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "extension": ".xlsx",
                "manual_download_required": True,
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
            },
            {
                "source_component": "policies",
                "raw_relpath": "data/raw/ccdf/policies/ccdf_policies_extract_2023.csv",
                "filename": "ccdf_policies_extract_2023.csv",
                "extension": ".csv",
                "manual_download_required": True,
                "landing_page": "https://ccdf.urban.org/search-database",
            },
        ]
    )


def ccdf_admin_long() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 1,
                "column_name": "state",
                "value_text": "California",
                "value_numeric": None,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 1,
                "column_name": "subsidized_private_slots",
                "value_text": "120000",
                "value_numeric": 120000.0,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 1,
                "column_name": "public_admin_slots",
                "value_text": "25200",
                "value_numeric": 25200.0,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 1,
                "column_name": "children_served_average_monthly",
                "value_text": "145200",
                "value_numeric": 145200.0,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 2,
                "column_name": "state",
                "value_text": "Texas",
                "value_numeric": None,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 2,
                "column_name": "children_served_average_monthly",
                "value_text": "98700",
                "value_numeric": 98700.0,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 2,
                "column_name": "subsidized_private_slots",
                "value_text": "80000",
                "value_numeric": 80000.0,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "admin",
                "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                "filename": "fy2023_table_1.xlsx",
                "file_format": "xlsx",
                "landing_page": "https://acf.gov/occ/data/child-care-and-development-fund-statistics",
                "source_sheet": "Table 1",
                "row_number": 2,
                "column_name": "public_admin_slots",
                "value_text": "18700",
                "value_numeric": 18700.0,
                "table_year": 2023,
                "parse_status": "parsed",
            },
        ]
    )


def ccdf_policy_long() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_component": "policies",
                "raw_relpath": "data/raw/ccdf/policies/ccdf_policies_extract_2023.csv",
                "filename": "ccdf_policies_extract_2023.csv",
                "file_format": "csv",
                "landing_page": "https://ccdf.urban.org/search-database",
                "source_sheet": "__default__",
                "row_number": 1,
                "column_name": "state",
                "value_text": "California",
                "value_numeric": None,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "policies",
                "raw_relpath": "data/raw/ccdf/policies/ccdf_policies_extract_2023.csv",
                "filename": "ccdf_policies_extract_2023.csv",
                "file_format": "csv",
                "landing_page": "https://ccdf.urban.org/search-database",
                "source_sheet": "__default__",
                "row_number": 1,
                "column_name": "copayment_required",
                "value_text": "yes",
                "value_numeric": None,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "policies",
                "raw_relpath": "data/raw/ccdf/policies/ccdf_policies_extract_2023.csv",
                "filename": "ccdf_policies_extract_2023.csv",
                "file_format": "csv",
                "landing_page": "https://ccdf.urban.org/search-database",
                "source_sheet": "__default__",
                "row_number": 2,
                "column_name": "state",
                "value_text": "Texas",
                "value_numeric": None,
                "table_year": 2023,
                "parse_status": "parsed",
            },
            {
                "source_component": "policies",
                "raw_relpath": "data/raw/ccdf/policies/ccdf_policies_extract_2023.csv",
                "filename": "ccdf_policies_extract_2023.csv",
                "file_format": "csv",
                "landing_page": "https://ccdf.urban.org/search-database",
                "source_sheet": "__default__",
                "row_number": 2,
                "column_name": "copayment_required",
                "value_text": "no",
                "value_numeric": None,
                "table_year": 2023,
                "parse_status": "parsed",
            },
        ]
    )


def sipp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2021,
                "subgroup": "all_ref_parents",
                "records": 1200,
                "weight_sum": 1500000.0,
                "any_paid_childcare_rate": 0.41,
                "avg_weekly_paid_childcare_all": 96.0,
                "avg_weekly_paid_childcare_payers": 234.0,
                "center_care_rate": 0.27,
                "family_daycare_rate": 0.08,
                "nonrelative_care_rate": 0.15,
                "head_start_rate": 0.03,
                "nursery_preschool_rate": 0.19,
                "geography": "national",
            },
            {
                "year": 2021,
                "subgroup": "under5_ref_parents",
                "records": 540,
                "weight_sum": 710000.0,
                "any_paid_childcare_rate": 0.53,
                "avg_weekly_paid_childcare_all": 142.0,
                "avg_weekly_paid_childcare_payers": 268.0,
                "center_care_rate": 0.34,
                "family_daycare_rate": 0.11,
                "nonrelative_care_rate": 0.18,
                "head_start_rate": 0.06,
                "nursery_preschool_rate": 0.25,
                "geography": "national",
            },
            {
                "year": 2023,
                "subgroup": "all_ref_parents",
                "records": 1260,
                "weight_sum": 1540000.0,
                "any_paid_childcare_rate": 0.44,
                "avg_weekly_paid_childcare_all": 108.0,
                "avg_weekly_paid_childcare_payers": 246.0,
                "center_care_rate": 0.29,
                "family_daycare_rate": 0.07,
                "nonrelative_care_rate": 0.16,
                "head_start_rate": 0.03,
                "nursery_preschool_rate": 0.21,
                "geography": "national",
            },
            {
                "year": 2023,
                "subgroup": "under5_ref_parents",
                "records": 580,
                "weight_sum": 745000.0,
                "any_paid_childcare_rate": 0.56,
                "avg_weekly_paid_childcare_all": 156.0,
                "avg_weekly_paid_childcare_payers": 279.0,
                "center_care_rate": 0.37,
                "family_daycare_rate": 0.10,
                "nonrelative_care_rate": 0.19,
                "head_start_rate": 0.07,
                "nursery_preschool_rate": 0.27,
                "geography": "national",
            },
        ]
    )


def ce() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2023,
                "subgroup": "with_children_u18",
                "records": 1400,
                "weight_sum": 1380000.0,
                "childcare_spender_rate": 0.17,
                "avg_childcare_spend_pq_all": 268.0,
                "avg_childcare_spend_pq_payers": 1575.0,
                "childcare_spend_share_pq": 0.014,
                "geography": "national",
            },
            {
                "year": 2023,
                "subgroup": "with_child_age_1_5",
                "records": 980,
                "weight_sum": 990000.0,
                "childcare_spender_rate": 0.22,
                "avg_childcare_spend_pq_all": 354.0,
                "avg_childcare_spend_pq_payers": 1655.0,
                "childcare_spend_share_pq": 0.019,
                "geography": "national",
            },
        ]
    )


def oews() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": "06", "year": 2021, "oews_childcare_worker_wage": 18.4, "oews_preschool_teacher_wage": 24.8, "oews_outside_option_wage": 21.2},
            {"state_fips": "06", "year": 2022, "oews_childcare_worker_wage": 19.0, "oews_preschool_teacher_wage": 25.4, "oews_outside_option_wage": 22.0},
            {"state_fips": "48", "year": 2021, "oews_childcare_worker_wage": 13.4, "oews_preschool_teacher_wage": 18.9, "oews_outside_option_wage": 16.8},
            {"state_fips": "48", "year": 2022, "oews_childcare_worker_wage": 13.9, "oews_preschool_teacher_wage": 19.3, "oews_outside_option_wage": 17.3},
            {"state_fips": "36", "year": 2021, "oews_childcare_worker_wage": 18.9, "oews_preschool_teacher_wage": 27.1, "oews_outside_option_wage": 22.8},
            {"state_fips": "36", "year": 2022, "oews_childcare_worker_wage": 19.5, "oews_preschool_teacher_wage": 27.8, "oews_outside_option_wage": 23.3},
        ]
    )


def noaa() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"county_fips": "06037", "state_fips": "06", "year": 2021, "storm_event_count": 12, "storm_property_damage": 85000.0, "storm_exposure": 0.24, "precip_event_days": 5, "source_url": "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2021_c20240620.csv.gz"},
            {"county_fips": "06073", "state_fips": "06", "year": 2021, "storm_event_count": 8, "storm_property_damage": 42000.0, "storm_exposure": 0.16, "precip_event_days": 3, "source_url": "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2021_c20240620.csv.gz"},
            {"county_fips": "48113", "state_fips": "48", "year": 2021, "storm_event_count": 32, "storm_property_damage": 520000.0, "storm_exposure": 0.64, "precip_event_days": 14, "source_url": "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2021_c20240620.csv.gz"},
            {"county_fips": "48201", "state_fips": "48", "year": 2021, "storm_event_count": 50, "storm_property_damage": 1200000.0, "storm_exposure": 1.00, "precip_event_days": 19, "source_url": "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2021_c20240620.csv.gz"},
            {"county_fips": "36061", "state_fips": "36", "year": 2021, "storm_event_count": 18, "storm_property_damage": 310000.0, "storm_exposure": 0.36, "precip_event_days": 8, "source_url": "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2021_c20240620.csv.gz"},
            {"county_fips": "36029", "state_fips": "36", "year": 2021, "storm_event_count": 6, "storm_property_damage": 15000.0, "storm_exposure": 0.12, "precip_event_days": 3, "source_url": "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2021_c20240620.csv.gz"},
        ]
    )


def ahs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"job_id": 1, "cbsa_code": "31080", "year": 2021, "job_type_code": "36", "job_type": "Added or replaced landscaping or sprinkler system", "job_group": "lot_yard_and_outbuildings", "job_cost": 2400, "job_diy": 0, "weight": 1.0, "housing_vintage": 1985, "home_value": 780000, "household_income": 98000, "tenure_owner": 1, "storm_exposure": 0.12},
            {"job_id": 2, "cbsa_code": "31080", "year": 2021, "job_type_code": "12", "job_type": "Remodeled bathroom", "job_group": "kitchen_bath_remodel", "job_cost": 7800, "job_diy": 1, "weight": 1.1, "housing_vintage": 1974, "home_value": 820000, "household_income": 110000, "tenure_owner": 1, "storm_exposure": 0.12},
            {"job_id": 3, "cbsa_code": "19100", "year": 2021, "job_type_code": "36", "job_type": "Added or replaced landscaping or sprinkler system", "job_group": "lot_yard_and_outbuildings", "job_cost": 1100, "job_diy": 0, "weight": 0.9, "housing_vintage": 1996, "home_value": 320000, "household_income": 72000, "tenure_owner": 1, "storm_exposure": 0.08},
            {"job_id": 4, "cbsa_code": "19100", "year": 2021, "job_type_code": "16", "job_type": "Added or replaced roofing", "job_group": "exterior_envelope", "job_cost": 1500, "job_diy": 1, "weight": 1.0, "housing_vintage": 1989, "home_value": 280000, "household_income": 68000, "tenure_owner": 1, "storm_exposure": 0.08},
            {"job_id": 5, "cbsa_code": "35620", "year": 2021, "job_type_code": "13", "job_type": "Remodeled kitchen", "job_group": "kitchen_bath_remodel", "job_cost": 9600, "job_diy": 0, "weight": 1.0, "housing_vintage": 1968, "home_value": 920000, "household_income": 125000, "tenure_owner": 1, "storm_exposure": 0.16},
            {"job_id": 6, "cbsa_code": "35620", "year": 2021, "job_type_code": "36", "job_type": "Added or replaced landscaping or sprinkler system", "job_group": "lot_yard_and_outbuildings", "job_cost": 1700, "job_diy": 1, "weight": 1.1, "housing_vintage": 1978, "home_value": 690000, "household_income": 103000, "tenure_owner": 1, "storm_exposure": 0.16},
        ]
    )
