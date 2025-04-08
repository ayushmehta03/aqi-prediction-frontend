import React from 'react';
import   icon from '../src/assets/searchbar.png' 
import menu from '../src/assets/Vector@2x.png'
import home from '../src/assets/home.png'
import air from '../src/assets/air.png'
import not from '../src/assets/notification.png'
import graph from '../src/assets/graph.png'
import solution from '../src/assets/solutions.png'
import Navbar from  '../Components/Navbar'
import Weather from '../Components/Weather';
import { Link } from 'react-router-dom';
const First = () => {



  return (
    <>
<div className="main1">
    <img src={menu} alt="" className="menu" />
    <img  className="home"src={home} alt="" />
    <div className="page-render">
  <Link  className='link2' to='/realdata' >< img className='menu' src={air} alt=""/> </Link>
    <Link className='link2' to='/graphs' > <img className='menu' src={graph} alt="" /> </Link>
    <Link className='link2'  to='/solutions'>  <img  className="menu"src={solution} alt="" /> </Link>
      <img  className='menu' src={not} alt="" />
    </div>
</div>
<Navbar 
image={icon}
/>
<Weather
/>


</>
  );
}

export default First;
