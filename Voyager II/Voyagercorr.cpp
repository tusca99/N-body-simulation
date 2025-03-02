#include <fstream>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <array>

struct p
{
    double m;
    double r[3], v[3];
};

struct pv
{
    double r[3], v[3];
};

p read(std::ifstream &file)
{
    p p;
    file >> p.m;
    file >> p.r[0] >> p.r[1] >> p.r[2];
    file >> p.v[0] >> p.v[1] >> p.v[2];
    for (size_t i = 0; i < 3; i++)
    {
        p.r[i] = p.r[i]*1000;
        p.v[i] = p.v[i]*1000;
    }
    return p;
}

std::vector<p> init(std::ifstream &file)
{
    int n;
    file >> n;
    std::vector<p> s;    
    for(size_t i=0; i<n; i++)
    {
        p p = read(file);
        s.push_back(p);
    }
    return s; 
}

std::vector<p> vecpmatch (std::vector<p> &s)
{
    std::vector<p> s1;
    for (size_t i = 0; i < s.size(); i++) s1.push_back(s[i]);
    return s1;
}

p centerm(std::vector<p> &s)
{
    p cm;
    for (size_t i = 0; i < s.size(); i++) cm.m += s[i].m;
    for (size_t i = 0; i < s.size(); i++)
    {
        for (size_t k = 0; k < 3; k++)
        {
            cm.r[k] += (s[i].m * s[i].r[k]);
            cm.v[k] += (s[i].m * s[i].v[k]);
        }
    }
    for (size_t k = 0; k < 3; k++)
    {
        cm.r[k] = cm.r[k]/cm.m;
        cm.v[k] = cm.v[k]/cm.m;
    }
    return cm;
}

std::array<double,3> forza(std::vector<p> &s, p &pi)
{
    const double G=6.67430e-11;
    std::array<double, 3> r = {0.0, 0.0, 0.0};
    std::array<double,3> forza = {0.0, 0.0, 0.0}; 
    for (size_t j = 0; j < s.size(); j++)
    {
        if(s[j].m != pi.m)
        { 
            for (size_t k = 0; k < 3; k++) r[k] = s[j].r[k] - pi.r[k];
            double modr = sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] );
            double f = (G * s[j].m * pi.m)/(modr*modr*modr);
            for (size_t k = 0; k < 3; k++) forza[k] += f*r[k];
        }
    }
   return forza;
}

std::vector<p> sistema_verlet(std::vector<p> &s, double dt)
{
    p cm=centerm(s);
    std::vector<std::array<double,3>> forze;
    for (size_t i = 0; i < s.size(); i++)
    {
        for (size_t k = 0; k < 3; k++)
        {
            s[i].v[k] -= cm.v[k];
        } 
        std::array<double,3> f=forza(s,s[i]);
        forze.push_back(f);
        for (size_t k = 0; k < 3; k++) s[i].r[k] = s[i].r[k] + s[i].v[k]*dt + (0.5/s[i].m)* f[k] *dt*dt; //r
    }    
    for (size_t i = 0; i < s.size(); i++)
    {
        std::array<double,3> f1=forza(s,s[i]);
        for (size_t k = 0; k < 3; k++) s[i].v[k] = s[i].v[k] + (0.5/s[i].m)*(forze[i][k] + f1[k])*dt; //v
    }
    return s;
}

std::vector<std::string> filenames(std::vector<p> &s)
{
    std::vector<std::string> fnames;
    for (int i = 0; i < s.size(); i++)
    {
        char buffer[30];
        std::sprintf(buffer, "object%d.dat" ,i);
        fnames.push_back(buffer);
    }
    return fnames;
}

void writeappfiles(std::vector<p> &s, std::vector<std::string> &fnames, double t) 
{
   for (int i = 0; i < s.size(); i++)
   {
        std::ofstream file;
        file.open(fnames[i], std::ios_base::app);
        file << t << " " << s[i].r[0] << " " << s[i].r[1] << " " << s[i].r[2] << " " << s[i].v[0] << " " << s[i].v[1] << " " << s[i].v[2] << " " << sqrt(s[i].r[0]*s[i].r[0] + s[i].r[1]*s[i].r[1] + s[i].r[2]*s[i].r[2]) << " " << sqrt(s[i].v[0]*s[i].v[0] + s[i].v[1]*s[i].v[1] + s[i].v[2]*s[i].v[2]) <<"\n";
        file.close();
   }
}

void checkremovefiles(std::vector<std::string> &fnames)
{
    for (size_t i = 0; i < fnames.size(); i++)
    {
        try
        {
            if ( std::filesystem::remove( fnames[i] ) ) std::cout << "file " << fnames[i] << " deleted.\n";
            else std::cout << "file " << fnames[i] << " not found.\n";
        }
        catch(const std::filesystem::filesystem_error& err) 
        {
            std::cout << "filesystem error: " << err.what() << '\n';
        }
    } 
}

bool checkvelocità(std::vector<p> &s1, std::vector<pv> &Vpv, int i, double step, double sv)
{
    bool check;
    if ( (fabs(s1[s1.size()-1].v[0] - Vpv[i/step].v[0])/(Vpv[i/step].v[0])) < sv &&
     (fabs(s1[s1.size()-1].v[1] - Vpv[i/step].v[1])/(Vpv[i/step].v[1])) < sv && 
     (fabs(s1[s1.size()-1].v[2] - Vpv[i/step].v[2])/(Vpv[i/step].v[2])) < sv ) 
     {
         check = true;
     }
    else check = false;
    return check;
}

double enpotcorpo(std::vector<p> &s, p&pi) 
{
    const double G=6.67430e-11;
    double r[3];
    double U = 0;
    for (size_t j = 0; j < s.size(); j++)
    {
        if(s[j].m != pi.m)
        { 
            for (size_t k = 0; k < 3; k++) r[k] = s[j].r[k] - pi.r[k];  
            double modr = sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] );
            U += -(G * s[j].m * pi.m)/(modr); 
        }
    }
   return U;
}

double encincorpo(p &pi) 
{
    double K = 0;
    double modv2 = ( pi.v[0]*pi.v[0] + pi.v[1]*pi.v[1] + pi.v[2]*pi.v[2] );
    K = (0.5 * pi.m)*(modv2); 
    return K;
}

int main(int argc, const char * argv[])
{
    //lettura dati iniziali
    std::ifstream file;
    file.open("sistema.dat", std::ios::in);
    std::vector<p> s = init(file);
    file.close();


    //impostazione parametri
    double giorno = 8.64e4;
    double step = 240;
    double dt = giorno/step;
    double anni = 20;
    long double t = giorno*365*anni; //tempo in anni
    long int n = t/dt;
    const double G=6.67430e-11;
    double sv = 0.003;
    bool correzione = true;
    
    //lettura dati nasa
    std::ifstream filenasa;
    filenasa.open("pvinnasa.dat", std::ios::in);
    std::vector<pv> Vpv;
    
    for (int i = 0; i < 365*anni; i++)
    {
        pv posvel;
        filenasa >> posvel.r[0] >> posvel.r[1] >> posvel.r[2]>> posvel.v[0] >> posvel.v[1] >> posvel.v[2];
        for (size_t k = 0; k < 3; k++)
        {
            posvel.r[k] *= 1000;
            posvel.v[k] *= 1000; // km/s --> SI
        }
        Vpv.push_back(posvel);
    }
    //cicli verlet
    auto fnames = filenames(s);
    checkremovefiles(fnames);
    std::cout<<"numero di cicli: "<< n <<"\n";
    std::cout<<"frequenza dei cicli: "<< dt <<" secondi"<<"\n";
    std::cout << "numero dati nasa: " << Vpv.size() << "\n";
    
    std::vector<p> s1;
    std::vector<double> pvswap;
    std::vector<double> Uv;
    std::vector<double> Kv;
    std::vector<double> Kvnasa;
    std::vector<double> Uvnasa;
    int printcounter = 0;
    writeappfiles (s, fnames, 0*dt);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        s1 = sistema_verlet(s,dt);
        if( printcounter == 1*step) 
        {
            writeappfiles (s1, fnames, i*dt);
            printcounter = 0;
            if (correzione == true)
            {
                if (checkvelocità(s1, Vpv, i, step, sv) == false) //3% di soglia rispetto alla v vera
                {
                    for (size_t k = 0; k < 3; k++) s1[s1.size()-1].v[k] = Vpv[i/step].v[k];
                    pvswap.push_back(i/step);
                } 
            }
            //energie del voyager II
            Uv.push_back(enpotcorpo(s1, s1[s1.size()-1]));
            Kv.push_back(encincorpo(s1[s1.size()-1]));
            
            Kvnasa.push_back( 0.5*s1[s1.size()-1].m*(Vpv[i/step].v[0]*Vpv[i/step].v[0] + Vpv[i/step].v[1]*Vpv[i/step].v[1] + Vpv[i/step].v[2]*Vpv[i/step].v[2]) );
            double Uvn = 0;
            for (size_t j = 0; j < s1.size(); j++)
            {
                double r[3];
                if(s[j].m != s1[s1.size()-1].m)
                { 
                    for (size_t k = 0; k < 3; k++) r[k] = s[j].r[k] - Vpv[i/step].r[k];  
                    double modr = sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] );
                    Uvn += -(G * s[j].m * s1[s1.size()-1].m)/(modr); 
                }
            }
            Uvnasa.push_back(Uvn);
        
        }
        s = vecpmatch(s1);
        printcounter += 1;
    }

    //stampo correzioni
    /*
    if (correzione == true)
    {
        std::ofstream swfile;
        swfile.open("dvswap.dat", std::ios::out);
        for (size_t i = 0; i < pvswap.size(); i++)
        {
            swfile << pvswap[i] <<"\n";
        }
        swfile.close();
        std::cout << "numero correzioni velocità: " << pvswap.size() << " con sv= "<< sv <<"\n";
    }

    //stampo energie voyager
    std::ofstream uvfile;
    uvfile.open("Ev.dat", std::ios::out);
    for (size_t i = 0; i < Uv.size(); i++)
    {
        uvfile << Kv[i] << " " << Uv[i] << " " << Kvnasa[i] << " " << Uvnasa[i] <<"\n";
    }
    uvfile.close();
    */
    auto current_time = std::chrono::high_resolution_clock::now();
    std::cout << "Program has been running for " << std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() << " seconds" << std::endl;
    return 0; 
}
//per sistema.dat: n=25: Sole,Mercurio,Venere,:0,1,2
//3:Terra,Luna,
//5:Marte,
//6:Giove,callisto,ganimede,europa,io,
//11:Saturno,iapetus,titano,rhea,dione,tethys,
//17:Urano,oberon,titania,umbriel,ariel,
//22:Nettuno,triton
//24:Voyager II
//coordinate in km,km/s --> SI
//compilare con -std=c++17 e chrono+filesystem, o g++-11 di default
