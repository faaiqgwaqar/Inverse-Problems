/*******************************************************************
    PT100RTD
   
    This library uses integer math to convert a Pt100 RTD resistance to a
    celsius temperature of two decimal point precision.
 
    It searches an ordered list of empirical resistances which correspond to
    temperatures ranging -200C to +850C, the complete span for which 100 ohm
    platinum RTDs are specified.  It interpolates temperatures from intermediate
    resistances.
 
    AUTHOR: drh
    DATE:2/18/2017
*******************************************************************/


#ifndef PT100RTD_H
#define PT100RTD_H

#include <stdint.h>
#include <math.h>

// refine conditional inclusion as per R.VanTwisk request.

#if (defined(__AVR__))
#include <avr\pgmspace.h>
#else
#include <pgmspace.h>
#endif

#if (ARDUINO >= 100)
  #include "Arduino.h"                  
#else
  #include "WProgram.h"
#endif



#define CELSIUS_MIN -200
#define CELSIUS_MAX 850

/*******************************************************************
*    pt100rtd list of resistances v temperature
*
*    DIN 43760 resistance values (ohms) were multiplied by 100 and
*    converted to 16 bit unsigned integers with no loss of accuracy.
* 
*    Examples:
* 1852 represents 18.52 ohms and corresponds to a temperature of -200C.
* 10000 ==> 100.00 ohms @   0C
* 13851 ==> 138.51 ohms @ 100C 
* 
*   The list resides in flash since it's too large for SRAM or EEPROM
*******************************************************************/

const static uint16_t Pt100_table[1051] PROGMEM  = {                         
  1852, 1896, 1939, 1982, 2025, 2068, 2111, 2154, 2197, 2240, 
  2283, 2326, 2369, 2412, 2455, 2497, 2539, 2582, 2625, 2667, 
  2710, 2752, 2795, 2837, 2880, 2922, 2965, 3007, 3049, 3092, 
  3134, 3176, 3218, 3261, 3303, 3345, 3386, 3428, 3470, 3512, 
  3554, 3596, 3638, 3680, 3722, 3763, 3805, 3847, 3889, 3931, 
  3972, 4014, 4056, 4097, 4139, 4180, 4222, 4264, 4305, 4346, 
  4388, 4429, 4471, 4512, 4553, 4595, 4635, 4676, 4718, 4759, 
  4800, 4841, 4882, 4923, 4964, 5006, 5047, 5088, 5129, 5170, 
  5211, 5252, 5292, 5333, 5374, 5415, 5456, 5497, 5538, 5578, 
  5619, 5660, 5700, 5741, 5782, 5822, 5863, 5904, 5944, 5985, 
  6026, 6067, 6107, 6148, 6187, 6229, 6269, 6310, 6350, 6391, 
  6430, 6470, 6511, 6551, 6591, 6631, 6672, 6712, 6752, 6792, 
  6833, 6873, 6913, 6953, 6993, 7033, 7073, 7113, 7153, 7193, 
  7233, 7273, 7313, 7353, 7393, 7433, 7473, 7513, 7553, 7593, 
  7633, 7673, 7713, 7752, 7792, 7832, 7872, 7911, 7951, 7991, 
  8031, 8070, 8110, 8150, 8189, 8229, 8269, 8308, 8348, 8388, 
  8427, 8467, 8506, 8546, 8585, 8625, 8664, 8704, 8743, 8783, 
  8822, 8862, 8901, 8940, 8980, 9019, 9059, 9098, 9137, 9177, 
  9216, 9255, 9295, 9334, 9373, 9412, 9452, 9491, 9530, 9569, 
  9609, 9648, 9687, 9726, 9765, 9804, 9844, 9883, 9922, 9961, 
  10000,10039,10078,10117,10156,10195,10234,10273,10312,10351,
  10390,10429,10468,10507,10546,10585,10624,10663,10702,10740,
  10779,10818,10857,10896,10935,10973,11012,11051,11090,11128,
  11167,11206,11245,11283,11322,11361,11399,11438,11477,11515,
  11554,11593,11631,11670,11708,11747,11785,11824,11862,11901,
  11940,11978,12016,12055,12093,12132,12170,12209,12247,12286,
  12324,12362,12401,12439,12477,12517,12555,12593,12632,12670,
  12708,12746,12785,12823,12861,12899,12938,12976,13014,13052,
  13090,13128,13167,13205,13243,13281,13319,13357,13395,13433,
  13471,13509,13547,13585,13623,13661,13699,13737,13775,13813,
  13851,13889,13927,13965,14003,14039,14077,14115,14153,14191,
  14229,14266,14304,14342,14380,14418,14456,14494,14532,14569,
  14607,14645,14682,14720,14758,14795,14833,14871,14908,14946,
  14983,15021,15058,15096,15134,15171,15209,15246,15284,15321,
  15358,15395,15432,15471,15508,15546,15583,15621,15658,15696,
  15733,15771,15808,15845,15883,15920,15956,15994,16031,16068,
  16105,16143,16180,16217,16254,16291,16328,16366,16403,16440,
  16477,16514,16551,16588,16625,16662,16700,16737,16774,16811,
  16848,16885,16922,16959,16996,17033,17069,17106,17143,17180,
  17217,17254,17291,17327,17364,17401,17439,17475,17512,17549,
  17586,17623,17659,17696,17733,17770,17806,17843,17880,17916,
  17953,17990,18026,18063,18099,18136,18173,18209,18246,18282,
  18319,18355,18392,18428,18465,18501,18538,18574,18611,18647,
  18684,18720,18756,18793,18829,18865,18902,18938,18974,19011,
  19047,19083,19120,19156,19192,19228,19266,19302,19338,19374,
  19410,19447,19483,19519,19555,19590,19626,19662,19698,19735,
  19771,19807,19843,19879,19915,19951,19987,20023,20059,20095,
  20131,20167,20203,20238,20274,20310,20346,20382,20418,20454,
  20490,20525,20561,20597,20633,20670,20705,20741,20777,20813,
  20848,20884,20920,20955,20991,21027,21062,21098,21134,21169,
  21205,21240,21276,21312,21347,21383,21419,21455,21490,21526,
  21561,21597,21632,21668,21703,21739,21773,21808,21844,21879,
  21915,21950,21985,22021,22056,22091,22127,22162,22197,22232,
  22268,22303,22338,22373,22409,22445,22480,22515,22550,22585,
  22621,22656,22691,22726,22761,22796,22831,22866,22901,22936,
  22972,23007,23042,23077,23112,23147,23181,23216,23251,23286,
  23321,23356,23391,23426,23460,23495,23530,23565,23600,23635,
  23670,23705,23740,23775,23809,23844,23879,23914,23948,23983,
  24018,24052,24087,24122,24156,24191,24225,24260,24295,24329,
  24364,24398,24433,24467,24502,24536,24571,24605,24640,24674,
  24709,24743,24778,24812,24846,24881,24915,24950,24984,25018,
  25053,25089,25121,25155,25190,25224,25259,25294,25328,25362,
  25396,25430,25465,25499,25533,25567,25601,25635,25670,25704,
  25738,25772,25806,25840,25874,25908,25942,25976,26010,26044,
  26078,26112,26146,26180,26214,26248,26283,26317,26350,26384,
  26418,26452,26486,26520,26554,26587,26621,26655,26689,26722,
  26756,26790,26824,26857,26891,26925,26958,26992,27026,27059,
  27093,27127,27160,27194,27227,27261,27295,27328,27362,27395,
  27429,27462,27496,27529,27563,27596,27631,27664,27697,27731,
  27764,27798,27831,27864,27898,27931,27964,27998,28031,28064,
  28098,28131,28164,28197,28231,28264,28297,28330,28363,28397,
  28430,28463,28496,28529,28562,28595,28630,28663,28696,28729,
  28762,28795,28828,28861,28894,28927,28960,28993,29026,29059,
  29092,29125,29158,29190,29223,29256,29290,29323,29356,29389,
  29421,29454,29487,29520,29553,29585,29618,29651,29684,29716,
  29749,29782,29814,29847,29880,29912,29945,29978,30010,30043,
  30075,30108,30141,30173,30206,30238,30271,30303,30336,30368,
  30401,30433,30466,30498,30530,30563,30595,30628,30660,30692,
  30725,30757,30789,30822,30854,30886,30919,30951,30983,31015,
  31048,31080,31112,31145,31178,31210,31243,31275,31307,31339,
  31371,31404,31436,31468,31500,31532,31564,31596,31628,31660,
  31692,31724,31756,31788,31820,31852,31885,31917,31949,31981,
  32012,32044,32076,32108,32140,32172,32203,32234,32266,32298,
  32330,32361,32393,32425,32457,32488,32521,32553,32585,32616,
  32648,32679,32711,32743,32774,32806,32838,32869,32901,32932,
  32964,32995,33027,33058,33090,33121,33153,33184,33216,33247,
  33279,33310,33341,33373,33404,33436,33468,33499,33531,33562,
  33593,33625,33656,33687,33718,33750,33781,33812,33843,33875,
  33906,33937,33968,33999,34030,34062,34094,34125,34155,34187,
  34218,34249,34280,34311,34342,34373,34404,34435,34466,34497,
  34528,34559,34590,34621,34652,34683,34715,34746,34776,34807,
  34838,34869,34900,34931,34961,34992,35023,35054,35085,35115,
  35146,35177,35207,35238,35269,35299,35330,35361,35391,35422,
  35453,35483,35514,35544,35575,35606,35637,35668,35698,35729,
  35759,35790,35820,35851,35881,35912,35942,35972,36003,36033,
  36064,36094,36124,36155,36185,36215,36246,36276,36306,36336,
  36367,36397,36427,36457,36488,36518,36549,36579,36609,36640,
  36670,36700,36730,36760,36790,36820,36850,36881,36911,36941,
  36971,37001,37031,37061,37091,37121,37152,37182,37212,37241,
  37271,37301,37331,37361,37391,37421,37451,37480,37510,37540,
  37570,37600,37629,37659,37689,37719,37749,37779,37809,37839,
  37868,37898,37928,37957,37987,38017,38046,38076,38105,38135,
  38165,38194,38224,38253,38283,38312,38342,38371,38401,38430,
  38460,38489,38518,38548,38577,38607,38637,38666,38696,38725,
  38755,38784,38813,38842,38872,38901,38931,38961,38990,39019,
  39048     // Pt100 resistance * 100 at 850C           
} ;                                                             

const int PT100_TABLE_MAXIDX = ((sizeof(Pt100_table) / sizeof(uint16_t)) - 1) ;

class pt100rtd
{
private:
    int search_pt100_list(uint16_t key) ;
    float ohmsX100_to_celsius(uint16_t ohmsX100) ;

public:
    pt100rtd() ;

    float celsius (uint16_t ohmsX100) ;
    float celsius(float Rrtd) ;

    float celsius_to_Pt100ohms (float celsius) ;

    float celsius_cvd(float R_ohms) ;
    float celsius_cubic(float R_ohms) ;
    float celsius_polynomial (float R_ohms) ;
    float celsius_rationalpolynomial (float R_ohms) ;
} ;

#endif  // PT100RTD_H

pt100rtd::pt100rtd() { ; }


/**********************************************************************
** Function Name:  search_pt100_list
**
** Description:   binary search
**    if match
**        return index of a match
**    if no match
**        return index of the smallest table value > key
** 
**  usually requires the maximum of log2(1051) probes, ==10, 
**  when search key is not an exact match.
**
**  Note: search must not return index == 0.
**  Calling function must exclude boundary cases
**  where (ohmsX100 <= table[0]).
** 
** Parameters:
**    uint16_t ohmsX100
**
** Uses:
** Returns: int index of nearest resistance value
** Creation: 1/26/2017 4:48a Daniel R. Haney
**********************************************************************/

int pt100rtd::search_pt100_list(uint16_t ohmsX100)
{
    int lower = 0 ;
    int upper = PT100_TABLE_MAXIDX ;
    int mid = (lower + upper) / 2 ;

    do
    {
  uint16_t pt100val = pgm_read_word_near(&Pt100_table[mid]) ;
    
  if (pt100val == ohmsX100)
  {
      break;
  }
  else if (pt100val < ohmsX100)
  {
      lower = mid + 1 ;
  }
  else
  {
      upper = mid ;
  }
    
  mid = (lower + upper) / 2 ;
    
    } while (lower < upper) ;
  // falls through on last mismatch

    return(mid);
}

/**********************************************************************
** Function Name: ohmsX100_to_celsius
**
** Description:
**  Look up (unsigned short int)(Pt100 resistance * 100) in table.
**    Interpolate temperature for intermediate resistances.
** 
**    Calling function must exclude boundary cases where
**    ohmsX100 <= table[0] && ohmsX100 >= table[MAX]
** 
** Parameters:
**  uint16_t Rrtd = 100 * (Pt100 RTD resistance in ohms)
**
** Uses:  Pt100_table
** Returns: float temperature celsius
**
** Creation: 1/26/2017 10:41a Daniel R. Haney
**********************************************************************/
float pt100rtd::ohmsX100_to_celsius (uint16_t ohmsX100)
{
    uint16_t R_upper, R_lower ;
    int hundredths = 0 ;    // STFU flag for avr-gcc
    int iTemp = 0 ;
    float celsius ;

    int index = search_pt100_list(ohmsX100) ;
  
    // The minimum integral temperature
    iTemp = index - 1 + CELSIUS_MIN ;
  
    // fetch floor() and ceiling() resistances since
    // key = intermediate value is the most likely case.

    // ACHTUNG!  (index == 0) is forbidden!
    R_lower = pgm_read_word_near(&Pt100_table[index - 1]) ;
    R_upper = pgm_read_word_near(&Pt100_table[index]) ;

    // if key == table entry, temp is an integer degree
    if (ohmsX100 == R_upper)
    {
  iTemp++ ;
  hundredths = 0 ;
    }
    // an intermediate resistance is the common case
    else if (ohmsX100 < R_upper)
    {
  hundredths = ((100 * (ohmsX100 - R_lower)) / (R_upper - R_lower)) ;
    }
    // two unlikely cases are included for disaster recovery
    else if (ohmsX100 > R_upper) /*NOTREACHED*/  /*...unless list search was dain bramaged */
    {
  iTemp++ ;
  // risks index+1 out of range
  uint16_t Rnext = pgm_read_word_near(&Pt100_table[index + 1]) ;
  hundredths = (100 * (ohmsX100 - R_upper)) / (Rnext - R_upper) ;
    }
    else  /*NOTREACHED*/  /*...except in cases of excessive tweakage at 2:30am */
    {
  hundredths = ((100 * (ohmsX100 - R_lower)) / (R_upper - R_lower)) ;
    }

    celsius  = (float)iTemp + (float)hundredths / 100.0 ;

    return(celsius );
}


/**********************************************************************
** Function Name: celsius (uint16_t)
** Function Name: celsius (float)
**
** Description:
**    return celsius temperature for a given Pt100 RTD resistance
**
**    This wrapper function excludes boundary cases where
**      ohmsX100 <= table[0] && ohmsX100 >= table[MAX]
** 
** Creation: 2/18/2017 2:29p Daniel R. Haney
**********************************************************************/

// Uses minimally-processed ADC binary output,
// an unsigned 16 bit integer == (ohms * 100).

float pt100rtd::celsius (uint16_t ohmsX100)
{
    // clip underflow 
    if (ohmsX100 <= pgm_read_word_near(&Pt100_table[0]))
    {
  // return min boundary temperature
  return((float) CELSIUS_MIN);
    }
    // clip overflow
    else if (ohmsX100 >= pgm_read_word_near(&Pt100_table[PT100_TABLE_MAXIDX]))
    {
  // return max boundary temperature
  return((float) CELSIUS_MAX);
    }
    else
    {
  return(pt100rtd::ohmsX100_to_celsius(ohmsX100)) ;
    }
}


// Uses a floating point resistance value.

float pt100rtd::celsius (float rtd_ohms)
{
    // convert to unsigned short
    uint16_t ohmsX100 = (uint16_t) floor(rtd_ohms * 100.0) ;
    return pt100rtd::celsius(ohmsX100) ;
}



/**********************************************************************
** Function Name:   celsius_to_Pt100ohms
**
** Description:     Return a Pt100 resistance corresponding to a temperature.
**        Seemed like a handy thing to have.
**
** Parameters:      float celsius
**
** Uses:      Pt100_table[], entries are uint16_t (ohms * 100)
** Returns:       float
** Creation: 2/24/2017 10:42a Daniel R. Haney
**********************************************************************/

float pt100rtd::celsius_to_Pt100ohms (float celsius)
{
    float Pt100_ohms, T_delta ;
    uint16_t R_lower, R_delta, R_fraction ;
    int upper, lower ;

    if (celsius < (float)CELSIUS_MIN)
    {
  R_lower = Pt100_table[0] ;
  R_fraction = 0 ;
    }
    else if (celsius > (float)CELSIUS_MAX)
    {
  R_lower = Pt100_table[PT100_TABLE_MAXIDX] ;
  R_fraction = 0 ;
    }
    else
    {
  lower = ((int) floor(celsius)) - CELSIUS_MIN ;
  upper = ((int) ceil(celsius)) - CELSIUS_MIN ; 

  R_lower = Pt100_table[lower] ;
  R_delta = Pt100_table[upper] - R_lower ;

  // fractional (non-integer) temperature
  T_delta = celsius - floor(celsius) ;

  // R_fraction is ohms * 100 (integer)
  // 0.5 is for round up before floor() truncation
  R_fraction = (uint16_t) floor(0.5 + (T_delta * (float)(R_delta))) ;
    }

    Pt100_ohms = ((float) (R_lower + R_fraction)) / 100.0 ;

    return(Pt100_ohms) ;
}


// inverse callendar van dusen formula.
// accurate from -60C up to 850 C.

#define PT100_NOMINAL 100.0
#define iCVD_A 3.9083e-3  // ganked from Adafruit_Max318656 library .h
#define iCVD_B -5.775e-7  // ditto.

float pt100rtd::celsius_cvd(float R_ohms)
{
    float Z1, Z2, Z3, Z4, temp;

    //Serial.print("Resistance: "); Serial.println(Rt, 8);

    Z1 = -iCVD_A;
    Z2 = iCVD_A * iCVD_A - (4 * iCVD_B);
    Z3 = (4 * iCVD_B) / PT100_NOMINAL;
    Z4 = 2 * iCVD_B;

    temp = Z2 + (Z3 * R_ohms);
    temp = (sqrt(temp) + Z1) / Z4;

    return (temp );
}


// cubic approximation
float pt100rtd::celsius_cubic(float R_ohms)
{
    float T = -247.29 + R_ohms * ( 2.3992 + R_ohms * (0.00063962 + 1.0241E-6 * R_ohms)) ;
    return(T );
}


// R2T polynomial from Analog Devices AN709 app note.
// implementation ganked from Adafruit MAX31865 library.
// Use for accurate temperatures -60C and below.
// Warning! Exceeds Class B tolerance spec above +164C

float pt100rtd::celsius_polynomial (float R_ohms)
{
    float rpoly, temp ;
    rpoly = R_ohms ;

    temp = -242.02 ;
    temp += 2.2228 * rpoly ;
    rpoly *= R_ohms ;     // square ;
    temp += 2.5859e-3 * rpoly ;
    rpoly *= R_ohms ;     // ^3 ;
    temp -= 4.8260e-6 * rpoly ;
    rpoly *= R_ohms ;     // ^4 ;
    temp -= 2.8183e-8 * rpoly ;
    rpoly *= R_ohms ;     // ^5 ;
    temp += 1.5243e-10 * rpoly ;

    return(temp);
}

// Rational polynomial fraction approximation taken from
// Mosaic Industries.com page on "RTD calibration."
// Accurate, probably beyond the ITS-90 spec
float pt100rtd::celsius_rationalpolynomial (float R_ohms)
{
    float num, denom, T ;

    float c0= -245.19 ;
    float c1 = 2.5293 ;
    float c2 = -0.066046 ;
    float c3 = 4.0422E-3 ;
    float c4 = -2.0697E-6 ;
    float c5 = -0.025422 ;
    float c6 = 1.6883E-3 ;
    float c7 = -1.3601E-6 ;

    num = R_ohms * (c1 + R_ohms * (c2 + R_ohms * (c3 + R_ohms * c4))) ;
    denom = 1.0 + R_ohms * (c5 + R_ohms * (c6 + R_ohms * c7)) ;
    T = c0 + (num / denom) ;

    return(T );
}

#include <Adafruit_MAX31865.h>

Adafruit_MAX31865 max = Adafruit_MAX31865(10, 11, 12, 13);
// use hardware SPI, just pass in the CS pin
//Adafruit_MAX31865 max = Adafruit_MAX31865(10);

// The value of the Rref resistor. Use 430.0!
#define RREF 4300.0

// Like, duh.
#define C2F(c) ((9 * c / 5) + 32)

// init the Pt100 table lookup module
pt100rtd PT100 = pt100rtd() ;

void setup()
{
  Serial.begin(115200) ;
  Serial.println("MAX31865 PT100 Sensor Test using NIST resistance table.");

  //max.begin(MAX31865_2WIRE);  // set to 2WIRE or 4WIRE as necessary
  //max.begin(MAX31865_3WIRE);  // set to 2WIRE or 4WIRE as necessary
  max.begin(MAX31865_3WIRE);  // set to 2WIRE or 4WIRE as necessary
}

void checkFault(void)
{
  // Check and print any faults
  uint8_t fault = max.readFault();
  if (fault)
  {
    Serial.print("Fault 0x"); Serial.println(fault, HEX);
    if (fault & MAX31865_FAULT_HIGHTHRESH)
    {
      Serial.println("RTD High Threshold"); 
    }
    if (fault & MAX31865_FAULT_LOWTHRESH)
    {
      Serial.println("RTD Low Threshold"); 
    }
    if (fault & MAX31865_FAULT_REFINLOW)
    {
      Serial.println("REFIN- > 0.85 x Bias"); 
    }
    if (fault & MAX31865_FAULT_REFINHIGH)
    {
      Serial.println("REFIN- < 0.85 x Bias - FORCE- open"); 
    }
    if (fault & MAX31865_FAULT_RTDINLOW)
    {
      Serial.println("RTDIN- < 0.85 x Bias - FORCE- open"); 
    }
    if (fault & MAX31865_FAULT_OVUV)
    {
      Serial.println("Under/Over voltage"); 
    }
    max.clearFault();
  }
}

void loop()
{
  uint16_t rtd, ohmsx100 ;
  uint32_t dummy ;
  float ohms, Tlut ;  
  float Tcvd, Tcube, Tpoly, Trpoly ;

  rtd = max.readRTD();

  // fast integer math:
  // fits in 32 bits as long as (100 * RREF) <= 2^16,
  //  i.e., RREF must not exceed 655.35 ohms (heh).
  // TO DO: revise for 4000 ohm reference resistor needed by Pt1000 RTDs
 
  // Use uint16_t (ohms * 100) since it matches data type in lookup table.
  dummy = ((uint32_t)(rtd << 1)) * 1000 * ((uint32_t) floor(RREF)) ;
  dummy >>= 16 ;
  ohmsx100 = (uint16_t) (dummy & 0xFFFF) ;

  // or use exact ohms floating point value.
  ohms = (float)(ohmsx100 / 1000) + ((float)(ohmsx100 % 1000) / 1000.0) ;

  Serial.print("rtd: 0x") ; Serial.print(rtd,HEX) ;
  Serial.print(", ohms: ") ; Serial.println(ohms,2) ;
 
  // compare lookup table and common computational methods
  
  Tlut  = PT100.celsius(ohmsx100) ;     // NoobNote: LUT== LookUp Table
  Tcvd  = PT100.celsius_cvd(ohms) ;         // Callendar-Van Dusen calc
  Tcube = PT100.celsius_cubic(ohms) ;       // Cubic eqn calc
  Tpoly = PT100.celsius_polynomial(ohms) ;        // 5th order polynomial
  Trpoly  = PT100.celsius_rationalpolynomial(ohms) ;  // ugly rational polynomial quotient
  
  // report temperatures at 0.001C resolution to highlight methodological differences
  Serial.print("Tlut   = ") ; Serial.print(Tlut  ,3) ; Serial.println(" C (exact)") ;
  Serial.print("Tcvd   = ") ; Serial.print(Tcvd  ,3) ; Serial.println(" C") ;
  Serial.print("Tcube  = ") ; Serial.print(Tcube ,3) ; Serial.println(" C") ;
  Serial.print("Tpoly  = ") ; Serial.print(Tpoly ,3) ; Serial.println(" C") ;
  Serial.print("Trpoly = ") ; Serial.print(Trpoly,3) ; Serial.println(" C") ;
    Serial.println();
  
  checkFault() ;

  delay(5000) ;

}
