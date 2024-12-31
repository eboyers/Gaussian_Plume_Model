using CSV, DataFrames

# filter data
data = CSV.read("/Users/ethanboyers/Desktop/gaussian_plume_dispersion_model/gp_model/hourly_ozone_data_2024.csv", DataFrame)
filtered_data = filter(row -> row["State Name"] == "California" && row["County Name"] == "Santa Clara", data)

filtered_df = DataFrame(
    date = filtered_data[:,"Date GMT"],
    time = filtered_data[:,"Time GMT"],
    latitude = filtered_data[:,"Latitude"],
    longitude = filtered_data[:,"Longitude"],
    measurements = filtered_data[:,"Sample Measurement"]
)

CSV.write("/Users/ethanboyers/Desktop/gaussian_plume_dispersion_model/gp_model/santa_clara_ozone.csv", filtered_df)
