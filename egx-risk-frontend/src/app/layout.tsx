import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Portfolio Risk Analytics — Omar Sherif",
  description: "Quantitative portfolio risk engine for the Egyptian Exchange (EGX).",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        {children}
        <span
          aria-hidden="true"
          className="pointer-events-none fixed bottom-6 right-6 z-[5] text-[10px] font-medium uppercase tracking-[0.2em] text-black opacity-60 dark:text-white"
        >
          Created by Omar Sherif
        </span>
      </body>
    </html>
  );
}
