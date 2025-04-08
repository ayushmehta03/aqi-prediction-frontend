import React from 'react';
import data from '../data';
import inhaler from "../src/assets/inhaler3.png"
import sinus from "../src/assets/sinus.png"
import flu from "../src/assets/flu.png"

const Solution = () => {
  




  return (
    


    <div>
      <h2 className='h1h2'>Precuation's based on our model</h2>
        <div className="main-solution">
            <div className="heading2">
            <p>AQI</p>
            <p>Cateogries</p>
            <p>Warnings</p>
            <p>Action</p>
           
            </div>
            <div className="solutions">
            <p className='para'>91.23</p>
            <p className='para'>Unhealthy for Sensitive Groups
            </p>
            <p className='para'>Wear masks outdoors</p>
            <p className='para'>Install air purifiers indoors</p>
            </div>
            <div   className="inf2">
            <p className='para'>211.31</p>
            <p className='para'>Hazardous
            </p>
            <p className='para'>Emergency conditions</p>
            <p className='para'>Use oxygen masks </p>
            </div>
            <div className="inf3">
              <p>127.36</p>
              <p>Unhealthy</p>
              <p>Wear N95 masks</p>
              <p>Sensitive groups should take care </p>
            </div>
            <div className="inf4">
              <p>64.54</p>
              <p>Moderate</p>
              <p>Air quality is acceptable</p>
              <p>Sensitive groups should take care </p>
            </div>
        </div>
        <div className="disease">
          <h3 className='major'>Major disease caused by High AQI</h3>
        <img src={inhaler} alt="" />
        <p className='head'>Asthama</p>
        <p className='desc'>Risk of Asthama symptoms is <span>high</span> when AQI is <span>Unhealthy(150-300)</span> </p>

        <img src={sinus} alt="" />
        <p className='head'>Sinus</p>
        <p className='desc'>Risk of Sinus symptoms is <span>high</span> when AQI is <span>Unhealthy(170-310)</span> </p>
        <img src={flu} alt="" />
        <p className='head'>Flu</p>
        <p className='desc'>Risk of Flu symptoms is <span>high</span> when AQI is <span>Unhealthy(190-340)</span> </p>

        </div>
    </div>
  );
}

export default Solution;
