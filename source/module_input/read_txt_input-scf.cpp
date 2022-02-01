//=======================
// AUTHOR : Daye Zheng
// DATE :   2022-01-10
//=======================

#include "read_txt_input_list.h"

#include "module_base/constants.h"
#include <stdexcept>

// for convert
#include "module_base/global_variable.h"
#include "../abacus_parameters.h"

namespace Read_Txt_Input
{
	void Input_List::set_items_scf()
	{
		this->output_labels.push_back("Parameters (05.SCF part)");

		{	// \sum |rhog_out - rhog_in |^2
			Input_Item item("scf_thr");
			item.default_1(1.0e-9);
			item.annotation = "charge density error";
			item.check_transform = [](Input_Item &self)
			{
				if(self.values[0].getd()<=0)
					throw std::invalid_argument("scf_thr must > 0");
			};
			item.convert = [](const Input_Item &self)
			{
				GlobalV::DRHO2 = self.values[0].getd();
				ABACUS::para.scfp.scf_thr = self.values[0].getd();
			};
			this->add_item(item);
		}

		/*{	// controling for choosing charge mixing method
			Input_Item item("mixing_type");
			item.default_1("pulay");
			item.annotation = "mixing method: plain, broyden, pulay, pulay-kerker, kerker";
			item.check_transform = [](Input_Item &self)
			{
				if(self.values[0].gets()=="plain")
				{
				}
				else if (self.values[0].gets()=="broyden" )
				{
				}
				else if (self.values[0].gets()=="pulay" )
				{
				}
				else if (self.values[0].gets()=="kerker" )
				{
				}
				else if (self.values[0].gets()=="pulay-kerker")
				{
				}
				else
					throw std::invalid_argument("mixing_type must be one of plain, broyden, pulay, pulay-kerker,
		kerker");
			};
			item.convert = [](const Input_Item &self)
			{
				//no convert
			};
			this->add_item(item);
		}*/

		{	// parameter in mixing methods
			Input_Item item("mixing_beta");
			item.default_1(0.7);
			item.annotation = "mixing parameter";
			item.check_transform = [](Input_Item &self)
			{
				if(self.values[0].getd()<=0)
					throw std::invalid_argument("mixing_beta must > 0");
			};
			item.convert = [](const Input_Item &self)
			{
				ABACUS::para.scfp.mixing_beta = self.values[0].getd();
			};
			this->add_item(item);
		}

		
	}
}