#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct p {
    double m;
    double r[3], v[3];
};
p read(std::ifstream &file) {
    p p;
    file >> p.m;
    file >> p.r[0] >> p.r[1] >> p.r[2];
    file >> p.v[0] >> p.v[1] >> p.v[2];
    for (size_t i = 0; i < 3; i++) {
        p.r[i] = p.r[i] * 1000;
        p.v[i] = p.v[i] * 1000;
    }
    return p;
}
std::vector<p> init(std::ifstream &file) {
    int n;
    file >> n;
    std::vector<p> s;
    for (size_t i = 0; i < n; i++) {
        p p = read(file);
        s.push_back(p);
    }
    return s;
}
std::vector<p> vecpmatch(std::vector<p> &s) {
    std::vector<p> s1;
    for (size_t i = 0; i < s.size(); i++)
        s1.push_back(s[i]);
    return s1;
}
p centerm(std::vector<p> &s) {
    p cm;
    for (size_t i = 0; i < s.size(); i++)
        cm.m += s[i].m;
    for (size_t i = 0; i < s.size(); i++) {
        for (size_t k = 0; k < 3; k++) {
            cm.r[k] += (s[i].m * s[i].r[k]);
            cm.v[k] += (s[i].m * s[i].v[k]);
        }
    }
    for (size_t k = 0; k < 3; k++) {
        cm.r[k] = cm.r[k] / cm.m;
        cm.v[k] = cm.v[k] / cm.m;
    }
    return cm;
}
std::array<double, 3> forza(std::vector<p> &s, p &pi) {
    const double G = 6.674e-11;
    double r[3]{0};
    // double vers[3];

    std::array<double, 3> forza{0};
    for (size_t j = 0; j < s.size(); j++) {
        if (s[j].m != pi.m) {
            for (size_t k = 0; k < 3; k++) {
                r[k] = s[j].r[k] - pi.r[k];
            }
            double modr = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            double f = (G * s[j].m * pi.m) / (modr * modr * modr);
            for (size_t k = 0; k < 3; k++)
                forza[k] += f * r[k];
        }
    }
    return forza;
}
double enpot(std::vector<p> &s, p &pi) //verificata con f_sole
{
    const double G = 6.674e-11;
    double r[3];
    double U;
    for (size_t j = 0; j < s.size(); j++) {
        if (s[0].m != pi.m) {
            for (size_t k = 0; k < 3; k++)
                r[k] = s[0].r[k] - pi.r[k];
            double modr = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            U = -(G * s[0].m * pi.m) / (modr);
        }
    }
    return U;
}
std::vector<p> sistema_verlet(std::vector<p> &s, double dt) {
    std::vector<p> s1 = vecpmatch(s);
    p cm = centerm(s);

    double(*forces)[3] = new double[s.size()][3];
    for (size_t i = 0; i < s.size(); i++) {
        // s1[i].m = s[i].m;
        // for (size_t k = 0; k < 3; k++)
        //     s[i].v[k] -= cm.v[k];

        std::array<double, 3> f = forza(s, s[i]);
        forces[i][0] = f[0];
        forces[i][1] = f[1];
        forces[i][2] = f[2];

        for (size_t k = 0; k < 3; k++)
            s1[i].r[k] = s[i].r[k] + s[i].v[k] * dt + (0.5 / s[i].m) * f[k] * dt * dt; //r
    }

    for (size_t i = 0; i < s.size(); i++) {
        std::array<double, 3> f1 = forza(s1, s1[i]);
        for (size_t k = 0; k < 3; k++)
            s1[i].v[k] = s[i].v[k] + (0.5 / s[i].m) * (forces[i][k] + f1[k]) * dt; //v
    }
    delete[] forces;
    return s1;
}
std::vector<std::string> filenames(std::vector<p> &s) {
    std::vector<std::string> fnames;
    for (int i = 0; i < s.size(); i++) {
        char buffer[30];
        std::sprintf(buffer, "object%d.dat", i);
        fnames.push_back(buffer);
    }
    return fnames;
}
void writeappfiles(std::vector<p> &s, std::vector<std::string> &fnames, double t) {
    for (int i = 0; i < s.size(); i++) {
        std::ofstream file;
        file.open(fnames[i], std::ios_base::app);
        file << t << "\t" << s[i].r[0] << "\t" << s[i].r[1] << "\t" << s[i].r[2] << "\n";
        file.close();
    }
}

int main(int argc, const char *argv[]) {
    std::ifstream file;
    file.open("sistema.dat", std::ios::in);
    std::vector<p> s = init(file);
    double giorno = 8.64e4;
    double dt = 100;
    // double anni = 1;
    // double t = giorno * 365 * anni; //tempo in anni
    double t = dt * 3e6;
    int n = t / dt;
    int every = 1e4;
    auto fnames = filenames(s);
    // checkremovefiles(fnames);
    std::cout << n << "\n";
    std::vector<p> s1;
    int printcounter = 0;
    // writeappfiles(s, fnames, 0 * dt);

    std::ofstream file_out;
    file_out.open("output.txt");
    file_out.precision(25);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {

        s1 = sistema_verlet(s, dt);

        if (i % every == 0) {
            file_out << i << "\t";
            for (size_t j = 0; j < s1.size(); j++) {
                file_out << s1[j].r[0] << "\t" << s1[j].r[1] << "\t";
            }
            file_out << std::endl;
        }

        // if (printcounter == 100 * 5) {
        //     writeappfiles(s1, fnames, i * dt);
        //     printcounter = 0;
        // }
        s = vecpmatch(s1);
        printcounter += 1;
        t += dt;
    }
    file_out.close();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::cout << "Program has been running for " << std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() << " seconds" << std::endl;
    return 0;
}
//per sistema.dat: n=10: Sole,Mercurio,Venere,Terra,Marte,Giove,Saturno,Urano,Nettuno,cometa di Halley
//coordinate in km,km/s --> SI