//=======================
// AUTHOR : Peize Lin
// DATE :   2021-12-13
//=======================

#include "read_txt_input_list.h"
#include "../module_base/tool_title.h"

namespace Read_Txt_Input
{
	void Input_List::add_item(const Input_Item &input_item)
	{
		this->list.insert(make_pair(input_item.label, input_item));
		this->output_labels.push_back(input_item.label);
	}

	/*
	void Input_List::check_value_size(const Input_Item& input_item, const int size)
	{
		if(input_item.value_read_size==-1)
			return;
		if(input_item.value_read_size!=size)
			throw std::out_of_range(std::to_string(input_item.value_read_size)+std::to_string(size));
	}	

	void Input_List::check_value_size(const Input_Item& input_item, const int size_low, const int size_up)
	{
		if(input_item.value_read_size==-1)
			return;
		if(input_item.value_read_size<size_low || input_item.value_read_size>size_up)
			throw std::out_of_range(std::to_string(input_item.value_read_size)+std::to_string(size_low)+std::to_string(size_up));
	}
	*/

	void Input_List::set_items()
	{
		ModuleBase::TITLE("Input_List","set_items");
		this->set_items_general();
		this->set_items_ipath();
		this->set_items_pw();
		this->set_items_lcao();
		this->set_items_scf();
		this->set_items_relax();
		this->set_items_print();
		this->set_items_efield();
		this->set_items_exx();
		this->set_items_md();
		this->set_items_dftu();
		this->set_items_vdw();
		this->set_items_berry();
		this->set_items_tddft();
		this->set_items_deepks();
		this->set_items_test();
		this->set_items_spectrum();
	}

}