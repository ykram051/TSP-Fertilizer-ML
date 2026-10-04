import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata={title:"TSP Process Intelligence",description:"Interactive process analytics, virtual sensing, forecasting and reporting for SLURRY_FREE_ACID."};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
