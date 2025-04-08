import React, { useState, useEffect } from "react";
import sun from "../src/assets/sun.png";
import moon from "../src/assets/moon.png";

const Greeting = () => {
  const [greeting, setGreeting] = useState("");

  useEffect(() => {
    const hours = new Date().getHours();
    if (hours < 12) {
      setGreeting("Good Morning");
    } else if (hours < 18) {
      setGreeting("Good Afternoon");
    } else {
      setGreeting("Good Evening");
    }
  }, []);

  return <span className="wish">{greeting}</span>;
};

// Navbar Component
const Navbar = ({ image, onSearch }) => {
  const [query, setQuery] = useState("");

  const handleKeyPress = (event) => {
    if (event.key === "Enter" && query.trim() !== "") {
      onSearch(query); // Call the function passed from Weather.jsx
      setQuery(""); // Clear input after search
    }
  };

  return (
    <div className="navbar">
      <div className="content">
        <span className="hello">Hello,</span>
        <Greeting />
      </div>
      <div className="part2">
        <div className="input">
          <img className="icon" src={image} alt="" />
          <input
            className="search"
            type="text"
            placeholder="Search your location"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress} // Detect Enter key
          />
        </div>
        <div className="button">
          <button className="first">
            <img src={sun} alt="" />
          </button>
          <button className="second">
            <img src={moon} alt="" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
