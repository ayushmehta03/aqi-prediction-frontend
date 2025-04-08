import React from 'react';
import carbon from '../src/assets/co21.png'
import lpg from '../src/assets/lpg2.png'
import nh3 from '../src/assets/nh3.png'
import ch4 from '../src/assets/ch4.png'
import smoke from '../src/assets/smoke.png'
import temp from '../src/assets/temp.png'
import rectangle1 from '../src/assets/rectangle1.png'
import rectangle2 from '../src/assets/rectangle2.png'
import rectangle3 from '../src/assets/rectangle3.png'
import rectangle4 from '../src/assets/rectangle4.png'


const Data = () => {
  return (
    <div className='real'>
      <h1>Real Time  <span>Tracking</span></h1>
      <div className="sensors-data">
        <div className="container">
        <img src={carbon}alt="" />
        <p className='car'>Carbon Monooxide</p>
        <p className='car2'><span>609</span> ppb</p>

        </div>
        <div className="container">
        <img src={lpg} alt="" />
        <p className="car car4">Liquid Petroleum Gas</p>
        <p className="car3"><span>208</span>ppb</p>
        
  </div>
        <div className="container">
    <img src={nh3} alt="" />
    <p className='car5'>Ammonia</p>
    <p className="car6"><span>40</span>ppm</p>

        </div>
        <div className="container">
            <img src={ch4}alt="" />
            <p className="car7">Methane</p>
            <p className="car8"><span>300</span>ppm</p>
        </div>
        <div className="container">
            <img src={smoke} alt="" />
            <p className='car9'>Smoke</p>
            <p className='car10'><span>120</span>ppm</p>
        </div>
        <div className="container">
            <img src={temp} alt="" />
            <p className='car11'>Temperature</p>
            <p className='car12'>16&deg;<span>C</span></p>
        </div>
      </div>
      <div className="side-rectangle">
    <div className="blue">
    <img src={rectangle1} alt="" />
    <span>Best</span>
    </div>
    <div className="voilet">
    <img src={rectangle2} alt="" />
    <span>Good</span>
    </div>
    <div className="purple">
    <img src={rectangle3} alt="" />
    <span>Need Attention</span>
    </div>
    <div className="aqua">
    <img src={rectangle4} alt="" />
    <span>Critical</span>
    </div>
   </div>

      </div>
      
    
  );
}

export default Data;
