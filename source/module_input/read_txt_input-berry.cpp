//=======================
// AUTHOR : Daye Zheng
// DATE :   2022-01-10
//=======================

#include "read_txt_input_list.h"

#include "module_base/constants.h"
#include <stdexcept>

// for convert
#include "module_base/global_variable.h"

namespace Read_Txt_Input
{
	void Input_List::set_items_berry()
	{
		this->output_labels.push_back("Parameters (13.Berry part)");

	}
}