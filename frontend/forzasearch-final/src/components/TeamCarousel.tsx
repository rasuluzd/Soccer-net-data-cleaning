"use client";

import Image from "next/image";

const TEAM = [
  { name: "Rasul Uzdijev", role: "Programming And Data Cleaning", image: "/team/Screenshot 2026-04-19 143937.png" },
  { name: "Thomas Knutsen", role: "Programming And Cleaning", image: "/team/Screenshot 2026-04-19 144450.png" },
  { name: "Abdi Sharif", role: "Front End Developer And Second In Command", image: "/team/Screenshot 2026-04-19 144022.png" },
  { name: "Tufa", role: "Document Writer And Scrum Master", image: null },
  { name: "Liban Hussein", role: "Lead Front End Developer And Group Leader", image: "/team/cv pic.jpg" },
];

export default function TeamCarousel() {
  return (
    <section className="min-h-screen pt-14 pb-6 bg-brand-cream dark:bg-brand-navy-light overflow-hidden">
      <div className="w-full mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <p className="text-sm uppercase tracking-[0.2em] text-gray-500 dark:text-white/40 mb-2">Meet The Team</p>
          <h2 className="font-display text-3xl md:text-4xl font-bold text-gray-900 dark:text-white">Behind The Project</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
          {TEAM.map((member) => (
            <div key={member.name} className="rounded-[2rem] bg-brand-navy text-white p-8 shadow-2xl border border-white/10 flex flex-col items-center justify-center text-center transition-transform duration-300 hover:-translate-y-1">
              <div className="w-24 h-24 rounded-full mb-6 flex items-center justify-center overflow-hidden">
                {member.image ? (
                  <Image
                    src={member.image}
                    alt={member.name}
                    width={96}
                    height={96}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-white/10 flex items-center justify-center text-brand-gold text-3xl font-semibold">
                    {member.name.split(" ").map((part) => part[0]).join("")}
                  </div>
                )}
              </div>
              <h3 className="text-xl font-semibold mb-3">{member.name}</h3>
              <p className="text-sm leading-relaxed text-white/70">{member.role}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
