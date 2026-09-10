import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { NavBar } from "@/components/NavBar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Causal Inference Agent",
  description:
    "Upload a dataset, ask a causal question in plain English, and get the design the data "
    + "can actually support — with its assumptions checked.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/*
          Apply the saved theme before first paint. Without this the page renders in the
          system theme and then flips, which is worse than a fractional delay here.
        */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark"){document.documentElement.setAttribute("data-theme",t)}}catch(e){}`,
          }}
        />
      </head>
      {/*
        Browser extensions (Grammarly, password managers) inject attributes onto <body>
        before React hydrates, which React reports as a hydration mismatch. This suppresses
        the warning for this element's own attributes only — it does not extend to the tree
        below, so genuine mismatches in the app still surface.
      */}
      <body className={`${inter.variable} font-sans antialiased`} suppressHydrationWarning>
        <NavBar />
        {children}
      </body>
    </html>
  );
}
