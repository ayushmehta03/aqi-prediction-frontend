import { useEffect, useState } from "react";
import React from "react";
import wind from "../src/assets/wind.png";
import visibility from "../src/assets/visibility.png";
import rise from "../src/assets/sunrise.png";
import set from "../src/assets/sunset.png";
import uv from "../src/assets/uv.png";
import humidity from "../src/assets/humidity.png";
import location from "../src/assets/location.png";
import rainycloud from "../src/assets/image 8.png";
import { Router,Link } from "react-router-dom";
const Weather = () => {
  const [city, setCity] = useState("Dehradun");
  const [weatherData, setWeatherData] = useState(null);
  const [timeNow, setTimeNow] = useState("");

  // Function to fetch weather data
  const search = async (city) => {
    try {
      const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${
        import.meta.env.VITE_APP_ID
      }&units=metric`;
      const response = await fetch(url);
      const data = await response.json();
      setWeatherData(data);
    } catch (error) {
      console.error("Error fetching weather data:", error);
    }
  };

  useEffect(() => {
    search("Dehradun");

    // Update time every second
    const interval = setInterval(() => {
      const today = new Date();
      const options = { hour: "2-digit", minute: "2-digit", hour12: true };
      setTimeNow(today.toLocaleTimeString("en-US", options));
    }, 1000);

    return () => clearInterval(interval); // Cleanup interval on unmount
  }, []);

  // Function to get current date
  const CurrentDate = () => {
    const today = new Date();
    const dayName = today.toLocaleDateString("en-US", { weekday: "long" });
    const day = today.getDate();
    const month = today.toLocaleString("en-US", { month: "short" });
    const year = today.getFullYear();
    const formattedDate = `${day}${month},${year}`;

    return (
      <div className="time">
        <p className="day">{dayName}</p>
        <p className="date">{formattedDate}</p>
      </div>
    );
  };

  return (
    <div>
      <div className="weather-main">
        <div className="location">
          <img src={location} alt="Location Icon" />
          <span>{city}</span>
        </div>
        <CurrentDate />
        <div className="temperature">
          <img src={rainycloud} alt="Weather Icon" />
        </div>
        <div className="feels">
          <p className="heavy">{weatherData?.weather?.[0]?.description}</p>
          <p className="deg">Feels like {Math.round(weatherData?.main?.feels_like)} &deg;</p>
        </div>
      </div>

      {/* Today's Highlights Section */}
      <div className="details">
        <span className="today-details">Today's Highlight</span>
        <div className="all-content">
          <div className="same">
            <div className="status">
              <img src={wind} alt="Wind Icon" />
              <span className="extra2">Wind Status</span>
            </div>
            <p className="extra">{weatherData?.wind?.speed} km/h</p>
            <span className="extra1">{timeNow}</span> {/* Correctly updating time */}
          </div>

          <div className="same">
            <div className="status">
              <img src={humidity} alt="Humidity Icon" />
              <span className="extra2">Humidity</span>
            </div>
            <p className="hum">{weatherData?.main?.humidity}%</p>
            <span className="humidity">Humidity is good</span>
          </div>

          <div className="different">
            <img src={rise} alt="Sunrise Icon" />
            <span>Sunrise</span>
            <p>{new Date(weatherData?.sys?.sunrise * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}</p>
          </div>

          <div className="same">
            <div className="status">
              <img src={uv} alt="UV Index Icon" />
              <span className="extra2">UV INDEX</span>
            </div>
            <p className="uv">4 UV</p>
            <span className="uvv">Moderate UV</span>
          </div>

          <div className="same">
            <div className="status">
              <img src={visibility} alt="Visibility Icon" />
              <span className="extra2">Visibility</span>
            </div>
            <p className="uv">{weatherData?.visibility / 1000} km</p>
            <span className="uvv">{timeNow}</span> {/* Correctly updating time */}
          </div>

          <div className="different">
            <img src={set} alt="Sunset Icon" />
            <span>Sunset</span>
            <p>{new Date(weatherData?.sys?.sunset * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}</p>
          </div>
        </div>
      </div>

      <div className="our-prediction">
        <p className="us">Our Solution </p>
      <div className="heading">
        <p>AQI</p>
        <p>Cateogry</p>
        <p>Warnings</p>
        <p className="action">Actions</p>
      </div>
      <div className="content1">
        <p>91.63</p>
        <p>Unhealthy</p>
        <p>Wear Masks</p>
        <p>Air Purifiers</p>
      </div>
      </div>
     <Link className="link-1"
     to="/solutions">
      Know More 
      </Link>
    </div>
  );
};

export default Weather;
