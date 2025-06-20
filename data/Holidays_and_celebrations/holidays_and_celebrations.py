import datetime
import pandas as pd
from dateutil.easter import easter
import holidays
import calendar
from pathlib import Path

def main():
    # Define the date range
    start_date = datetime.date(2011, 1, 1)
    end_date   = datetime.date(2026, 5, 31)
    years = list(range(start_date.year, end_date.year + 1))

    # Collect UK Public Holidays
    uk_holidays = holidays.UnitedKingdom(years=years)

    public_holidays = []
    for date, name in uk_holidays.items():
        if start_date <= date <= end_date:
            public_holidays.append({
                "date": date,
                "type": "Public Holiday",
                "name": name
            })

    # Collect other celebrations (non-holidays)
    other_events = []

    for year in years:
        # Valentine's Day — February 14
        other_events.append({
            "date": datetime.date(year, 2, 14),
            "type": "Celebration",
            "name": "Valentine's Day"
        })
        # Halloween — October 31
        other_events.append({
            "date": datetime.date(year, 10, 31),
            "type": "Celebration",
            "name": "Halloween"
        })
        # Bonfire Night — November 5
        other_events.append({
            "date": datetime.date(year, 11, 5),
            "type": "Celebration",
            "name": "Bonfire Night"
        })
        # Mother’s Day (UK) — Fourth Sunday in Lent = Easter Sunday − 21 days
        mday = easter(year) - datetime.timedelta(days=21)
        other_events.append({
            "date": mday,
            "type": "Celebration",
            "name": "Mother's Day"
        })
        # Father’s Day (UK) — Third Sunday in June
        june_days = [datetime.date(year, 6, d) for d in range(1, 31)]
        sundays_in_june = [d for d in june_days if d.weekday() == 6]
        fday = sundays_in_june[2]  # [2] = third Sunday
        other_events.append({
            "date": fday,
            "type": "Celebration",
            "name": "Father's Day"
        })

    # Collect major sporting events
    sport_events = []

    for year in years:
        # London Marathon — Third Sunday in April
        april_days = [datetime.date(year, 4, d) for d in range(1, 31)]
        sundays_in_april = [d for d in april_days if d.weekday() == 6]
        marathon = sundays_in_april[2]
        sport_events.append({
            "date": marathon,
            "type": "Sporting Event",
            "name": "London Marathon"
        })

        # Wimbledon Final — Second Saturday & Sunday in July
        july_days = [datetime.date(year, 7, d) for d in range(1, 32)]
        saturdays_in_july = [d for d in july_days if d.weekday() == 5]
        wim_sat = saturdays_in_july[1]
        wim_sun = wim_sat + datetime.timedelta(days=1)
        sport_events.append({
            "date": wim_sat,
            "type": "Sporting Event",
            "name": "Wimbledon Final (Saturday)"
        })
        sport_events.append({
            "date": wim_sun,
            "type": "Sporting Event",
            "name": "Wimbledon Final (Sunday)"
        })

        # FA Cup Final — Last Saturday in May
        may_days = [datetime.date(year, 5, d) for d in range(1, 32)]
        saturdays_in_may = [d for d in may_days if d.weekday() == 5]
        fa_cup = saturdays_in_may[-1]
        sport_events.append({
            "date": fa_cup,
            "type": "Sporting Event",
            "name": "FA Cup Final"
        })

            # Grand National — First Saturday in April
        april_days_full = [datetime.date(year, 4, d) for d in range(1, 31)]
        saturdays_in_april_full = [d for d in april_days_full if d.weekday() == 5]
        if saturdays_in_april_full:
            grand_national = saturdays_in_april_full[0]
            sport_events.append({
                "date": grand_national,
                "type": "Sporting Event",
                "name": "Grand National"
            })

        # FA Community Shield — First Saturday in August
        aug_days = [datetime.date(year, 8, d) for d in range(1, 32)]
        saturdays_in_aug = [d for d in aug_days if d.weekday() == 5]
        if saturdays_in_aug:
            fa_community = saturdays_in_aug[0]
            sport_events.append({
                "date": fa_community,
                "type": "Sporting Event",
                "name": "FA Community Shield"
            })

        # EFL (Carabao) Cup Final — Last Sunday in February
        feb_days = [datetime.date(year, 2, d) for d in range(1, 29 + (1 if calendar.isleap(year) else 0))]
        sundays_in_feb = [d for d in feb_days if d.weekday() == 6]
        if sundays_in_feb:
            efl_cup = sundays_in_feb[-1]
            sport_events.append({
                "date": efl_cup,
                "type": "Sporting Event",
                "name": "EFL (Carabao) Cup Final"
            })

        # UEFA Champions League Final — Last Saturday in May
        saturdays_in_may = [d for d in may_days if d.weekday() == 5]
        if saturdays_in_may:
            ucl_final = saturdays_in_may[-1]
            sport_events.append({
                "date": ucl_final,
                "type": "Sporting Event",
                "name": "UEFA Champions League Final"
            })

    # Add UEFA European Championship finals and FIFA World Cup finals
    extra_football_finals = [
        # Euros
        {"date": datetime.date(2012, 7, 1),  "name": "UEFA Euro 2012 Final"},
        {"date": datetime.date(2016, 7, 10), "name": "UEFA Euro 2016 Final"},
        {"date": datetime.date(2021, 7, 11), "name": "UEFA Euro 2020 Final"},
        # World Cups
        {"date": datetime.date(2014, 7, 13), "name": "FIFA World Cup 2014 Final"},
        {"date": datetime.date(2018, 7, 15), "name": "FIFA World Cup 2018 Final"},
        {"date": datetime.date(2022, 12, 18),"name": "FIFA World Cup 2022 Final"},
    ]

    for ev in extra_football_finals:
        if start_date <= ev["date"] <= end_date:
            sport_events.append({
                "date": ev["date"],
                "type": "Sporting Event",
                "name": ev["name"]
            })

    cricket_finals = [
        # ICC Men’s Cricket World Cup Finals
        {"date": datetime.date(2011, 4, 2),  "name": "CWC 2011 Final"},
        {"date": datetime.date(2015, 3, 29), "name": "CWC 2015 Final"},
        {"date": datetime.date(2019, 7, 14), "name": "CWC 2019 Final"},
        {"date": datetime.date(2023, 11, 19),"name": "CWC 2023 Final"},
    ]

    for ev in cricket_finals:
        if start_date <= ev["date"] <= end_date:
            sport_events.append({
                "date": ev["date"],
                "type": "Sporting Event",
                "name": ev["name"]
            })

    # Combine all events into one DataFrame
    events = pd.DataFrame(public_holidays + other_events + sport_events)

    # Filter to our exact date window (2011-01-01 to 2026-05-31)
    events = events[(events["date"] >= start_date) & (events["date"] <= end_date)].copy()

    # Add Year and Month columns for easy grouping
    events["date"] = pd.to_datetime(events["date"])
    events["Year"]  = events["date"].dt.year
    events["Month"] = events["date"].dt.month

    # Count how many of each event type occur in each (Year, Month)
    counts = (
        events
        .groupby(["Year", "Month", "type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # If any event-type columns are missing for some months, add them with zeros
    for col in ["Public Holiday", "Celebration", "Sporting Event"]:
        if col not in counts.columns:
            counts[col] = 0

    # Merge the counts with the baseline dataset
    data_dir = Path(__file__).resolve().parent.parent
    baseline_file = data_dir / 'Base/baseline_dataset.csv'
    baseline = pd.read_csv(baseline_file)
    baseline = baseline[["LSOA code 2021", "Year", "Month"]].drop_duplicates()

    final = pd.merge(baseline, counts, on=["Year", "Month"], how="left")
    final.fillna(0, inplace=True)

    # Save the result
    final.to_csv(data_dir / "Holidays_and_celebrations/holidays_finalized.csv", index=False)
    print(final.head())

# # List all event dates so you know exactly which days were used
# print("\n--- All Event Dates Considered ---")
# for _, row in events.sort_values("date").iterrows():
#     print(f"{row['date']}: {row['type']} — {row['name']}")