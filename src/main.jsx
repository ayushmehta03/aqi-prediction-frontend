import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App.jsx";
import Solution from "../Pages/Solution.jsx";
import Data from "../Pages/Data.jsx";
import Graphs from "../Pages/Graphs.jsx";
const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  {
    path: "/solutions",
    element: <Solution />,
  },
  {
    path: "/realdata",
    element: <Data />,
  },
  {
    path: "/graphs",
    element: <Graphs />,
  },

]);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
