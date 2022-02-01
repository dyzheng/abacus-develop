//=======================
// AUTHOR : Peize Lin
// DATE :   2021-12-13
//=======================

#ifndef READ_TXT_INIPUT_LIST_H
#define READ_TXT_INIPUT_LIST_H

#include "read_txt_input_item.h"

#include <vector>
#include <map>
#include <string>

namespace Read_Txt_Input
{
	class Input_List
	{
	public:
		void set_items();

	private:	
		void add_item(const Input_Item &input_item);
		//static void check_value_size(const Input_Item& input_item, const int size);
		//static void check_value_size(const Input_Item& input_item, const int size_low, const int size_up);

		std::map<std::string, Input_Item> list;
		std::vector<std::string> output_labels;

		///
		///01. all general input parameters for ABACUS
		///
		void set_items_general();
		///
		///02. all parameters for paths of input files
		///
		void set_items_ipath();
		///
		///03. all parameters for plane wave module
		///
		void set_items_pw();
		///
		///04. all parameters for lcao base only
		///
		void set_items_lcao();
		///
		///05. all parameters for self-consistant function 
		///
		void set_items_scf();
		///
		///06. all parameters for ionic or cell relaxation
		///
		void set_items_relax();
		///
		///07. all parameters for printed files controling
		///
		void set_items_print();
		///
		///08. all parameters for electric field
		///
		void set_items_efield();
		///
		///09. all parameters for exact exchange functional
		///
		void set_items_exx();
		///
		///10. all parameters for molecular dynamics
		///
		void set_items_md();
		///
		///11. all parameters for LDA plus U effect
		///
		void set_items_dftu();
		///
		///12. all parameters for Van Der Waals effect
		///
		void set_items_vdw();
		///
		///13. all parameters for berry curvature calculation
		///
		void set_items_berry();
		///
		///14. all parameters for time-dependent DFT
		///
		void set_items_tddft();
		///
		///15. all parameters for DeePKS
		///
		void set_items_deepks();
		///
		///16. all parameters for test only
		///
		void set_items_test();
		///
		///17. all parameters for spectrum calculation 
		///
		void set_items_spectrum();

		friend class Input_Process;
	};
}

#endif